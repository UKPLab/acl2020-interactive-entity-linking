from tqdm import tqdm

from gleipnir.formats import Corpus, DataPoint, convert_to_dataset
from gleipnir.kb import KnowledgeBase, KbHandle
from gleipnir.ranking import *


def convert_corpus_to_svm_light_training(corpus: Corpus, kb: KnowledgeBase, name: str):
    data = convert_to_dataset(corpus)

    with open(f"quickrank_{name}.txt", "w") as f:
        for i, entity in enumerate(tqdm(data)):
            qid = i + 1
            gold = kb.get_by_iri(entity.label)
            add_line(f, 1, qid, entity, gold, f"Gold: {entity.mention} - {gold.label if gold else 'NIL'} - {entity.label}")

            for candidate in  kb.search_mention(entity.mention):
                # Do not add gold as wrong candidate
                if candidate.uri == entity.label:
                    continue

                add_line(f, 0, qid, entity, candidate, f"Bad: {entity.mention} - {candidate.label} - {candidate.uri}")


def add_line(f, score: int, qid: int, entity: DataPoint, candidate: KbHandle, comment: str) -> str:
    features = [f"{i+1}:{score}" for i, score in enumerate(compute_features(entity, candidate))]
    line = f"{score} qid:{qid} {' '.join(features)} # {comment}"
    f.write(line)
    f.write("\n")


def compute_features(entity: DataPoint, candidate: KbHandle):
    return [
        levenshtein_mention_label(entity, candidate),
        levenshtein_context_description(entity, candidate),
        jaro_winkler_mention_label(entity, candidate),
        jaro_winkler_context_description(entity, candidate),
        exact_match_mention_label(entity, candidate),
        soundex_exact_match_mention_label(entity, candidate)
    ]



