import os

import attr

from sklearn import metrics

from gleipnir.formats import write_conllel, Corpus
from gleipnir.config import PATH_RESULTS


@attr.s
class EvaluationResult:
    micro_accuracy = attr.ib()      # type: float
    micro_precision = attr.ib()     # type: float
    micro_recall = attr.ib()        # type: float
    micro_f1 = attr.ib()            # type: float


def evaluate(corpus_gold: Corpus, corpus_predicted: Corpus) -> EvaluationResult:
    y_true = []
    y_pred = []

    for entity_gold, entity_predicted in zip(corpus_gold.iter_entities(), corpus_predicted.iter_entities()):
        assert entity_gold.start == entity_predicted.start
        assert entity_gold.end == entity_predicted.end

        y_true.append(entity_gold.uri)
        y_pred.append(entity_predicted.uri)

    micro_accuracy = metrics.accuracy_score(y_true, y_pred)
    micro_precision = metrics.precision_score(y_true, y_pred, average="micro")
    micro_recall = metrics.recall_score(y_true, y_pred, average="micro")
    micro_f1 = metrics.f1_score(y_true, y_pred, average="micro")

    result = EvaluationResult(
        micro_accuracy=micro_accuracy,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1
    )

    return result


def write_result(path_to_dataset: str, corpus: Corpus):
    dataset_name = os.path.basename(os.path.normpath(path_to_dataset))
    target_path = os.path.join(PATH_RESULTS, dataset_name)

    write_conllel(target_path, corpus)


