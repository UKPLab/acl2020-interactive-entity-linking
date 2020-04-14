import logging
from http import HTTPStatus
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from ariadne.classifier import Classifier
from ariadne.protocol import TrainingDocument
from ariadne.server import Server
from ariadne.util import setup_logging
from ariadne.contrib.stringmatcher import LevenshteinStringMatcher
from cassis import Cas, load_typesystem, load_cas_from_xmi
from flask import request, jsonify

from gleipnir.config import PATH_GENERATED, PATH_RECOMMENDER_MODELS
from gleipnir.handcrafted import FeatureGenerator, Feature
from gleipnir.models.letor_models import LetorModel
from gleipnir.models.ranksvm import SklearnRankSvmModel
from gleipnir.models.baselines import NoopBaseline

logger = logging.getLogger(__name__)


def handle_rerank_request(ranker: "RankingEntityLinker"):
    rerank_result = ranker.rerank(request.json)

    if rerank_result:
        ranks, explanations = rerank_result
        result = {
            "ranks": [int(x) for x in ranks],
            "explanations": explanations
        }

        return jsonify(result)
    else:
        return "Model not ready yet!", HTTPStatus.PRECONDITION_FAILED.value


class RankingEntityLinker(Classifier):

    def __init__(self):
        super().__init__(model_directory=Path(PATH_RECOMMENDER_MODELS))

        self.fg = FeatureGenerator()

    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id: str, user_id: str):
        data = []
        for doc in documents:
            data.extend(featurize_cas(self.fg, doc.cas))

        logger.info("Data len: %d", len(data))

        if len(data) == 0:
            logger.info("Not enough training data!")
            return

        df = pd.DataFrame(data)
        df.ext.name = "inception"

        if project_id.endswith("1") or project_id.endswith("3"):
            ModelClass = SklearnRankSvmModel
            print("Using hitl")
        else:
            ModelClass = NoopBaseline
            print("Using baseline")

        model = ModelClass()
        result = model.fit_evaluate(df, df)
        logger.debug("Result: %s", result)

        self._save_model(user_id, model)

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        pass

    def rerank(self, json_data: Dict[str, Any]) -> Tuple[List[int], List[dict]]:
        user_id = json_data["user"]
        model: LetorModel = self._load_model(user_id)

        if model is None:
            logger.info("Model not trained yet!")
            return []

        df = featurize_ranking_request(self.fg, json_data)

        if df is not None:
            ranks = model.rank(df)

            explanations = []

            feature_names = df.ext.features
            for _, row in df.iterrows():
                explanation = {f[5:]: row[f] for f in feature_names}
                explanations.append(explanation)

            return ranks, explanations
        else:
            return [], []


def featurize_cas(fg: FeatureGenerator, cas: Cas) -> List:
    features = get_features()

    results = []

    for qid, entity in enumerate(cas.select("webanno.custom.EntityLinking")):
        candidates = list(cas.select_covered("inception.internal.KbHandle", entity))

        if len(candidates) == 0:
            continue

        for i, candidate in enumerate(candidates):
            if entity.iri == candidate.iri:
                gold_idx = i
                break
        else:
            continue

        sentences = list(cas.select_covering("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence", entity))
        assert len(sentences) == 1
        sentence = sentences[0]

        mention = entity.get_covered_text().lower()
        context = sentence.get_covered_text().lower()
        l = len(context)
        # context = context[int(l * 0.25):int(l * 0.75)]

        for cid, candidate in enumerate(candidates):
            score = float(entity.iri == candidate.iri)
            query = candidate.query
            label = candidate.label.lower()

            result = fg.featurize_candidate(qid, cid, "inception_rank", score, mention, context,
                                            label or "", candidate.description or "",
                                            entity.iri, gold_idx,
                                            candidate.iri, features)

            result.update(fg.featurize_query(mention, query, label))



            results.append(result)

    return results


def featurize_ranking_request(fg: FeatureGenerator, json_data: Dict) -> Optional[pd.DataFrame]:
    if not json_data["candidates"]:
        logger.info("No candidates given to rerank!")
        return None

    features = get_features()

    mention = json_data["mention"].lower()
    context = json_data["context"].lower()
    query = json_data["query"].lower()

    print("Query", query)

    results = []
    for cid, candidate in enumerate(json_data["candidates"]):
        label = candidate["label"] or ''
        description = candidate["description"] or ''
        iri = candidate["iri"]

        label = label.lower()
        description = description.lower()

        result = fg.featurize_candidate(0, cid, "inception_rank", None, mention, context,
                                        label or "", description or "",
                                        None, None,
                                        iri, features)
        result.update(fg.featurize_query(mention, query, label))
        results.append(result)

    df = pd.DataFrame(results)
    df.ext.name = "inception"
    return df

def convert_stuff():
    with open(PATH_GENERATED + "/userstudy/obama/TypeSystem.xml", "rb") as f:
        typesystem = load_typesystem(f)

    with open(PATH_GENERATED + "/userstudy/obama/Wikipedia-Obama.xmi", "rb") as f:
        cas = load_cas_from_xmi(f, typesystem)

    featurize_cas(cas)


def build_recommender():
    ranker = RankingEntityLinker()

    server = Server()
    server.add_classifier("reranker", ranker)
    server.add_classifier("leven", LevenshteinStringMatcher())
    server._app.add_url_rule("/rank", "rank", lambda: handle_rerank_request(ranker), methods=["POST"])

    return server


def get_features() -> List[Feature]:
    features = list(Feature)
    features.remove(Feature.SENTENCE_BERT_CD)
    return features


if __name__ == '__main__':
    setup_logging()
    server = build_recommender()
    server.start(debug=True)
    # convert_stuff()

