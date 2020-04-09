import statistics
from typing import List

import numpy as np

import attr

from sklearn.metrics import average_precision_score


@attr.s
class EvaluationResult:
    accuracy: float = attr.ib()
    accuracy_top5: float = attr.ib()
    mean_reciprocal_rank: float = attr.ib()
    mean_number_of_candidates: float = attr.ib()
    gold_not_in_candidates: float = attr.ib()


def compute_letor_scores(grouped_labels: List[List[str]], grouped_scores: List[List[float]], gold_targets: List[str]) -> EvaluationResult:
    assert len(grouped_labels) == len(grouped_scores) == len(gold_targets)

    accuracy_buffer = []
    top_5_buffer = []
    mrr_buffer = []
    number_of_candidates_buffer = []
    gold_not_in_candidates_buffer = []

    for scores, labels, gold in zip(grouped_scores, grouped_labels, gold_targets):
        assert len(scores) == len(labels)

        sorted_indices = np.argsort(scores)[::-1]
        sorted_labels = list(labels[sorted_indices])
        prediction = sorted_labels[0]

        try:
            gold_idx_in_prediction = sorted_labels.index(gold) + 1
            mrr = 1.0 / gold_idx_in_prediction
        except ValueError:
            mrr = 0.0

        assert len(sorted_labels[:5]) <= 5

        accuracy_buffer.append(1.0 if prediction == gold else 0.0)
        top_5_buffer.append(1.0 if gold in sorted_labels[:5] else 0.0)
        mrr_buffer.append(mrr)
        number_of_candidates_buffer.append(len(labels))
        gold_not_in_candidates_buffer.append(1.0 if gold not in labels else 0.0)

    accuracy = statistics.mean(accuracy_buffer)
    accuracy_top5 = statistics.mean(top_5_buffer)
    mean_reciprocal_rank = statistics.mean(mrr_buffer)
    mean_number_of_candidates = statistics.mean(number_of_candidates_buffer)
    gold_not_in_candidates = statistics.mean(gold_not_in_candidates_buffer)

    # mean_average_precision = sum(map_buffer) / len(map_buffer)

    return EvaluationResult(accuracy=accuracy, accuracy_top5=accuracy_top5, mean_reciprocal_rank=mean_reciprocal_rank,
                            mean_number_of_candidates =mean_number_of_candidates, gold_not_in_candidates=gold_not_in_candidates)
