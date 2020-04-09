import logging
from enum import Enum

import unicodedata
from typing import Any

import pandas as pd

import jellyfish
import scipy

import textdistance
from sentence_transformers import SentenceTransformer

from tqdm import tqdm

from gleipnir.data import *
from gleipnir.formats import Document, Sentence, Entity
from gleipnir.kb import *
from gleipnir.config import PATH_HANDCRAFTED
from gleipnir.datasets import HandcraftedExtensionAccessor, get_raw_corpus_data

DEBUG = False

import nltk
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))


class Feature(Enum):
    # Surface form / Entity label based
    EXACT_MATCH_ML = "exact match ML"
    LABEL_IS_PREFIX_OF_MENTION = "label is prefix of mention"
    LABEL_IS_POSTFIX_OF_MENTION = "label is postfix of mention"
    MENTION_IS_PREFIX_OF_LABEL = "mention is prefix of label"
    MENTION_IS_POSTFIX_OF_LABEL = "mention is postfix of label"

    LABEL_IS_IN_MENTION = "label is in mention"
    MENTION_IS_IN_LABEL = "mention is in label"
    #LCSSEQ_ML = "longest common subsequence similarity ML"
    #LCSSTR_ML = "longest common substring similarity ML"
    #RATCLIFF_OBERSHELP_ML = "Ratcliff-Obershelp similarity ML"

    LEVENSHTEIN_ML = "levenshtein ML"
    JARO_WINKLER_ML = "jaro winkler ML"

    # Description based
    LEVENSHTEIN_CD = "levenshtein CD"
    JARO_WINKLER_CD = "jaro winkler CD"
    SORENSEN_DICE_CD = "sorensen dice CD"
    JACCARD_CD = "jaccard CD"
    #LCSSEQ_CD = "longest common subsequence similarity CD"
    #LCSSTR_CD = "longest common substring similarity CD"
    #RATCLIFF_OBERSHELP_CD = "Ratcliff-Obershelp similarity CD"

    # Phonetics based
    SOUNDEX_EXACT_MATCH_ML = "soundex exact match ML"
    MRA_ML = "mra ML"

    # Popularity based
    #POPULARITY = "popularity"
    #POPULARITY_WIKIDATA_ID = "wikidata popularity prior"

    # embedding based
    #BYTEPAIR_COSINE_ML = "bytepair ML"
    #BERT_COSINE_CD = "bert CD"

    # BERT
    SENTENCE_BERT_CD = "sentence bert CD"


class FeatureGenerator:

    def __init__(self):
        self._feature_to_func = {
            # Surface form / Entity label based
            Feature.EXACT_MATCH_ML: self._exact_match_mention_label,
            Feature.MENTION_IS_PREFIX_OF_LABEL: self._mention_is_prefix_of_label,
            Feature.MENTION_IS_POSTFIX_OF_LABEL: self._mention_is_postfix_of_label,
            Feature.LABEL_IS_PREFIX_OF_MENTION: self._label_is_prefix_of_mention,
            Feature.LABEL_IS_POSTFIX_OF_MENTION: self._label_is_postfix_of_mention,

            Feature.LABEL_IS_IN_MENTION: self._label_is_in_mention,
            Feature.MENTION_IS_IN_LABEL: self._mention_is_in_label,

            Feature.LEVENSHTEIN_ML: self._levenshtein_mention_label,
            Feature.JARO_WINKLER_ML: self._jaro_winkler_mention_label,
            Feature.SOUNDEX_EXACT_MATCH_ML: self._soundex_exact_match_mention_label,
            Feature.MRA_ML: self._mra_mention_label,

            # Feature.LCSSEQ_ML: self._lcsseq_ml,
            # Feature.LCSSTR_ML: self._lcsstr_ml,
            # Feature.RATCLIFF_OBERSHELP_ML: self._ratcliff_obershelp_ml,

            # Context / description based
            Feature.LEVENSHTEIN_CD: self._levenshtein_context_description,
            Feature.JARO_WINKLER_CD: self._jaro_winkler_context_description,
            Feature.SORENSEN_DICE_CD: self._sorensen_dice_context_description,
            Feature.JACCARD_CD: self._jaccard_context_description,

            # Feature.LCSSEQ_CD: self._lcsseq_ml,
            # Feature.LCSSTR_CD: self._lcsstr_ml,
            # Feature.RATCLIFF_OBERSHELP_CD: self._ratcliff_obershelp_ml,

            # Embedding based
            Feature.SENTENCE_BERT_CD: self._sentencebert_cd
        }

        self._sentence_bert = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
        self._sentence_bert_cache = Cache(os.path.join(CACHE, "sentence_bert_cache"))

    def compute_features_for_corpus(self, name: str, corpus: Corpus, cg: CandidateGenerator,
                                    desired_features: List[Feature], force_gold: bool) -> pd.DataFrame:
        logging.info("Concerting: [{}] for full", name)
        results = []

        qid = 0

        candidate_id = 0
        for doc_id, document in enumerate(tqdm(corpus.documents)):
            for sentence in document.sentences:
                for entity in sentence.entities.values():
                    mention = sentence.get_covered_text(entity)
                    context = sentence.get_text().lower()

                    gold = cg.get_gold_entity(entity.uri)

                    gold_idx = -1
                    if force_gold:
                        gold_idx = 0
                        candidates = cg.generate_negative_train_candidates(entity.uri, mention)

                        # If we generate a training set and have no negative candidates, we just skip the whole entity
                        # because we cannot generate a positive/negative pair from this later anyways
                        if gold == NIL and len(candidates) == 1 and candidates[0] == NIL:
                            continue

                        result = self.featurize_candidate(qid, candidate_id, document.name, 1.0, mention, context,
                                                          gold.label, gold.description,
                                                          gold.uri, gold_idx,
                                                          gold.uri, desired_features)

                        results.append(result)
                        candidate_id += 1
                    else:
                        candidates = cg.generate_test_candidates(mention)
                        for i, candidate in enumerate(candidates):
                            if gold.uri == candidate.uri:
                                gold_idx = i
                                break

                    # Remove potential duplicate NIL
                    if gold == NIL and NIL in candidates:
                        candidates.remove(NIL)

                    for candidate in candidates:
                        score = float(gold.uri == candidate.uri)
                        result = self.featurize_candidate(qid, candidate_id,document.name, score, mention, context,
                                                          candidate.label, candidate.description,
                                                          gold.uri, gold_idx,
                                                          candidate.uri, desired_features)

                        results.append(result)
                        candidate_id += 1

                    qid += 1

            if DEBUG:
                break

        df = pd.DataFrame(results)
        df.ext.name = name
        return df

    def compute_features_for_simulation(self, name: str, corpus: Corpus, kb: KnowledgeBase, desired_features: List[Feature]) -> pd.DataFrame:
        logging.info("Converting: [{}] for simulation", name)
        results = []

        qid = 0
        cid = 0

        # The annotation of a single item is done in three steps:
        # 1. Search only for the mention. If the gold is in the candidate list, pick it.
        # 2. Search for the beginning of the gold label
        # 3. Search for the whole gold label, this must return a candidate list that contains the final gold
        for doc_id, document in enumerate(tqdm(corpus.documents)):
            for sentence in document.sentences:
                for entity in sentence.entities.values():
                    mention = sentence.get_covered_text(entity)
                    context = sentence.get_text().lower()

                    gold_iri = entity.uri
                    gold_candidate = kb.get_by_iri(gold_iri)
                    gold_found = False

                    idx = 0

                    candidates = set()

                    # 1. Search only for the mention
                    mention_candidates = kb.search_mention(mention)
                    candidates.update(mention_candidates)

                    for candidate in mention_candidates:
                        if candidate.uri == gold_iri:
                            gold_found = True
                            phase = 1

                        idx += 1

                    # 2. Search for the beginning of the mention
                    if not gold_found:
                        user_query_candidates = kb.search_user_query(mention.split()[0]) - candidates
                        candidates.update(user_query_candidates)

                        for candidate in user_query_candidates:
                            if candidate.uri == gold_iri:
                                gold_found = True
                                phase = 2
                            idx += 1

                    # 3. Search for the whole gold label, this must return a candidate list that contains the final gold
                    if not gold_found:
                        if gold_candidate.label:
                            gold_query_candidates = kb.search_mention(gold_candidate.label.lower()) - candidates
                        else:
                            gold_query_candidates = set()

                        gold_query_candidates.add(gold_candidate)

                        candidates.update(gold_query_candidates)

                        for candidate in gold_query_candidates:
                            if candidate.uri == gold_iri:
                                gold_found = True
                                phase = 3

                            idx += 1

                    assert gold_found

                    candidates = list(candidates)

                    gold_idx = -1
                    for i, candidate in enumerate(candidates):
                        if candidate.uri == gold_iri:
                            gold_idx = i
                            break

                    for candidate in candidates:
                        score = float(gold_iri == candidate.uri)
                        result = self.featurize_candidate(qid, cid, "simulation", score, mention, context,
                                                        candidate.label or "", candidate.description or "",
                                                        gold_iri, gold_idx,
                                                        candidate.uri, desired_features)
                        result["phase"] = phase
                        results.append(result)

                    qid += 1

        df = pd.DataFrame(results)
        df.ext.name = name
        return df


    def featurize_candidate(self, qid: int, cid: int, document_name: str, score: float,
                             mention: str, context: str, label: str, description: str,
                             gold_uri: str, gold_idx: int, candidate_uri: str,
                             desired_features: List[Feature]) -> Dict[str, Any]:
        result = {}

        mention = mention.lower()
        context = context.lower()
        label = label.lower()
        description = description.lower()

        result["qid"] = qid
        result["score"] = score
        result["mention"] = mention
        result["context"] = context
        result["label"] = label
        result["description"] = description

        result["doc_id"] = document_name
        result["candidate_id"] = cid
        result["uri"] = candidate_uri
        result["gold"] = gold_uri
        result["gold_idx"] = gold_idx

        # context = " ".join([w for w in context.lower().split() if w not in STOPWORDS])
        # description = " ".join([w for w in description.lower().split() if w not in STOPWORDS])

        for feature in desired_features:
            feature_fn = self._feature_to_func[feature]
            result["feat_" + feature.value] = feature_fn(mention, context, label, description)

        return result

    def featurize_query(self, mention: str, query: str, label: str) -> Dict[str, float]:
        mention = mention.lower().strip()
        query = query.lower().strip()
        label = label.lower()

        if not query:
            return {
                "feat_query is prefix of mention": 0.0,
                "feat_query is prefix of label": 0.0,
                "feat_query is postfix of mention": 0.0,
                "feat_query is postfix of label": 0.0,
                "feat_query_in_label": 0.0,
                "feat_query_in_mention": 0.0,

                "feat_query_is_label": 0.0,

                "feat_levenshtein_QL": 1.0,
                "feat_levenshtein_QM": 1.0,
                "feat_jaro_QL": 1.0,
                "feat_jaro_QM": 1.0,

                "feat_jaccard_QL": 1.0,
                "feat_sorensen_dice_QL": 1.0,
            }

        result = {
            "feat_query is prefix of mention": float(mention.startswith(query)),
            "feat_query is prefix of label": float(label.startswith(query)),
            "feat_query is postfix of mention": float(mention.endswith(query)),
            "feat_query is postfix of label": float(label.endswith(query)),
            "feat_query_in_label": float(query in label),
            "feat_query_in_mention": float(query in mention),

            "feat_query_is_label": float(query == label),

            "feat_levenshtein_QL": textdistance.levenshtein.normalized_distance(query, label),
            "feat_levenshtein_QM": textdistance.levenshtein.normalized_distance(query, mention),
            "feat_jaro_QL": textdistance.jaro_winkler.normalized_distance(query, label),
            "feat_jaro_QM": textdistance.jaro_winkler.normalized_distance(query, mention),

            "feat_jaccard_QL": textdistance.jaccard.normalized_distance(query.split(), label.split()),
            "feat_sorensen_dice_QL": textdistance.sorensen_dice.normalized_distance(query.split(), label.split())
        }
        return result

    def _jaccard_context_description(self, mention: str, context: str, label: str, description: str) -> float:
        return textdistance.jaccard.normalized_distance(context.lower().split(), description.lower().split())
        return result

    # Feature computing functions

    def _length_mention(self, mention: str, context: str, label: str, description: str) -> float:
        return len(mention)

    def _length_label(self, mention: str, context: str, label: str, description: str) -> float:
        return len(label)

    def _length_context(self, mention: str, context: str, label: str, description: str) -> float:
        return len(context)

    def _length_description(self, mention: str, context: str, label: str, description: str) -> float:
        return len(description)

    def _exact_match_mention_label(self, mention: str, context: str, label: str, description: str) -> float:
        return float(label.lower() == mention.lower())

    def _mention_is_prefix_of_label(self, mention: str, context: str, label: str, description: str) -> float:
        return float(label.startswith(mention))

    def _mention_is_postfix_of_label(self, mention: str, context: str, label: str, description: str) -> float:
        return float(label.endswith(mention))

    def _label_is_prefix_of_mention(self, mention: str, context: str, label: str, description: str) -> float:
        return float(mention.startswith(label))

    def _label_is_postfix_of_mention(self, mention: str, context: str, label: str, description: str) -> float:
            return float(mention.endswith(label))

    def _label_is_in_mention(self, mention: str, context: str, label: str, description: str) -> float:
        return float(label in mention)

    def _mention_is_in_label(self, mention: str, context: str, label: str, description: str) -> float:
        return float(mention in label)

    def _levenshtein_mention_label(self, mention: str, context: str, label: str, description: str) -> float:
        return textdistance.levenshtein.normalized_distance(mention, label)

    def _jaro_winkler_mention_label(self, mention: str, context: str, label: str, description: str) -> float:
        result = textdistance.jaro_winkler.normalized_distance(mention.lower(), label)
        return result

    def _soundex_exact_match_mention_label(self, mention: str, context: str, label: str, description: str) -> float:
        s1 = jellyfish.soundex(self._normalize(mention).replace(" ", ""))
        s2 = jellyfish.soundex(self._normalize(label).replace(" ", ""))
        return float(s1.lower() == s2.lower())

    def _mra_mention_label(self, mention: str, context: str, label: str, description: str) -> float:
        return textdistance.mra.normalized_distance(mention, label)

    def _lcsseq_ml(self, mention: str, context: str, label: str, description: str) -> float:
        return textdistance.lcsseq.normalized_distance(mention.lower(), label)

    def _lcsstr_ml(self, mention: str, context: str, label: str, description: str) -> float:
        return textdistance.lcsstr.normalized_distance(mention.lower(), label)

    def _ratcliff_obershelp_ml(self, mention: str, context: str, label: str, description: str) -> float:
        return textdistance.ratcliff_obershelp.normalized_distance(mention.lower(), label)

    def _levenshtein_context_description(self, mention: str, context: str, label: str, description: str) -> float:
        return textdistance.levenshtein.normalized_distance(context.lower(), description)

    def _jaro_winkler_context_description(self, mention: str, context: str, label: str, description: str) -> float:
        return textdistance.jaro_winkler.normalized_distance(context.lower(), description)

    def _sorensen_dice_context_description(self, mention: str, context: str, label: str, description: str) -> float:
        return textdistance.sorensen_dice.normalized_distance(context.lower().split(), description.split())

    def _jaccard_context_description(self, mention: str, context: str, label: str, description: str) -> float:
        return textdistance.jaccard.normalized_distance(context.lower().split(), description.lower().split())

    def _lcsseq_cd(self, mention: str, context: str, label: str, description: str) -> float:
        return textdistance.lcsseq.normalized_distance(context.lower(), description.lower())

    def _lcsstr_cd(self, mention: str, context: str, label: str, description: str) -> float:
        return textdistance.lcsstr.normalized_distance(context.lower(), description.lower())

    def _ratcliff_obershelp_cd(self, mention: str, context: str, label: str, description: str) -> float:
        return textdistance.ratcliff_obershelp.normalized_distance(context.lower(), description.lower())

    def _sentencebert_cd(self, mention: str, context: str, label: str, description: str) -> float:
        # Context
        if context in self._sentence_bert_cache:
            context_embedding = self._sentence_bert_cache[context]
        else:
            context_embedding = self._sentence_bert.encode([context])
            self._sentence_bert_cache[context] = context_embedding

        # Description
        if description in self._sentence_bert_cache:
            description_embedding = self._sentence_bert_cache[description]
        else:
            description_embedding = self._sentence_bert.encode([description])
            self._sentence_bert_cache[description] = description_embedding

        return scipy.spatial.distance.cosine(context_embedding, description_embedding)

    # Util

    def _normalize(self, s: str) -> str:
        return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode("utf8")


def get_kb(kb_name: str, cached: bool = True) -> KnowledgeBase:
    if kb_name == "wwo-fuseki":
        return FusekiKnowledgeBase("wwo", caching=cached)
    elif kb_name == "1641-fuseki":
        return FusekiKnowledgeBase("depositions", caching=cached)
    elif kb_name == "aida":
        return WikidataVirtuosoKnowledgeBase(caching=cached)
    else:
        raise ValueError(f"Invalid model name: {kb_name}")

def main():
    os.makedirs(PATH_HANDCRAFTED, exist_ok=True)

    names = [
        "aida",
        "wwo-fuseki",
        "1641-fuseki",
    ]

    for name in names:
        data = get_raw_corpus_data(name)
        kb = get_kb(name, cached=True)
        features = list(Feature)
        # features = [Feature.LENGTH_CONTEXT]
        fg = FeatureGenerator()

        # df_train = fg.compute_features_for_corpus(name + "_train", data.corpus_train, data.cg, features, True)
        # df_dev = fg.compute_features_for_corpus(name + "_dev", data.corpus_dev, data.cg, features, True)
        # df_test = fg.compute_features_for_corpus(name + "_test", data.corpus_test, data.cg, features, True)
        # df_all = fg.compute_features_for_corpus(name + "_all", data.corpus_all, data.cg, features, True)

        # df_train.ext.to_csv()
        # df_dev.ext.to_csv()
        # df_test.ext.to_csv()
        # df_all.ext.to_csv()

        df_sim = fg.compute_features_for_simulation(name + "_full_sim", data.corpus_all, kb, features)
        df_sim.ext.to_csv()


if __name__ == '__main__':
    main()
