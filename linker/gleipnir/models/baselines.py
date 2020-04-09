from collections import defaultdict
import random
from typing import List

import numpy as np
import pandas as pd

from gleipnir.evaluation.metrics import EvaluationResult, compute_letor_scores
from gleipnir.formats import Corpus, DataPoint
from gleipnir.models.letor_models import LetorModel

from scipy import stats

class DictionaryBaseline(LetorModel):

    def __init__(self):
        super().__init__()
        self._model = None

    def fit_evaluate(self, df_train: pd.DataFrame, df_val: pd.DataFrame, **kwargs) -> EvaluationResult:
        model = defaultdict(lambda: defaultdict(int))
        mentions = df_train.ext.mentions
        uris = df_train.ext.gold_uris

        assert len(mentions) == len(uris), "Mentions and uris have to be of same length!"

        for mention, uri in zip(mentions, uris):
            model[mention][uri] += 1

        self._model = model
        return self.evaluate(df_val)

    def evaluate(self, df: pd.DataFrame) -> EvaluationResult:
        grouped_uris = []
        grouped_scores = []
        targets = df.ext.gold_uris

        for group in df.ext.groupby_qid:
            scores = self._get_scores(group)
            grouped_scores.append(scores)
            grouped_uris.append(group.ext.uris)

        return compute_letor_scores(grouped_uris, grouped_scores, targets)

    def rank(self, group: pd.DataFrame) -> List[int]:
        scores = self._get_scores(group)
        idx = np.argsort(scores)[::-1]
        return list(idx)

    def _get_scores(self, group: pd.DataFrame) -> List[float]:
        mention = group.ext.mentions[0]
        candidates = self._model[mention]

        if candidates:
            # Just use the entity that was most often linked with this mention
            prediction = max(candidates, key=candidates.get)
        else:
            prediction = "NIL"

        scores = []
        for uri in group.ext.uris:
            scores.append(float(uri == prediction))

        return scores


class RandomBaseline(LetorModel):

    def fit_evaluate(self, df_train: pd.DataFrame, df_val: pd.DataFrame, **kwargs) -> EvaluationResult:
        return self.evaluate(df_val)

    def evaluate(self, df: pd.DataFrame) -> EvaluationResult:
        grouped_uris = []
        grouped_scores = []
        targets = df.ext.gold_uris

        for group in df.ext.groupby_qid:
            scores = self._get_scores(group)
            grouped_scores.append(scores)
            grouped_uris.append(group.ext.uris)

        return compute_letor_scores(grouped_uris, grouped_scores, targets)

    def rank(self, group: pd.DataFrame) -> List[int]:
        scores = self._get_scores(group)
        idx = np.argsort(scores)[::-1]
        return list(idx)

    def _get_scores(self, group: pd.DataFrame) -> List[float]:
        return list(np.random.uniform(size=len(group)))


class LevenshteinBaseline(LetorModel):

    def fit_evaluate(self, df_train: pd.DataFrame, df_val: pd.DataFrame, **kwargs) -> EvaluationResult:
        return self.evaluate(df_val)

    def evaluate(self, df: pd.DataFrame) -> EvaluationResult:
        grouped_uris = []
        grouped_scores = []
        targets = df.ext.gold_uris

        for group in df.ext.groupby_qid:
            scores = self._get_scores(group)
            grouped_scores.append(scores)
            grouped_uris.append(group.ext.uris)

        return compute_letor_scores(grouped_uris, grouped_scores, targets)

    def rank(self, group: pd.DataFrame) -> List[int]:
        scores = self._get_scores(group)
        idx = np.argsort(scores)[::-1]
        return list(idx)

    def _get_scores(self, group: pd.DataFrame) -> List[float]:
        return list(group["feat_levenshtein ML"])


class NoopBaseline(LetorModel):

    def fit_evaluate(self, df_train: pd.DataFrame, df_val: pd.DataFrame, **kwargs) -> EvaluationResult:
        return self.evaluate(df_val)

    def evaluate(self, df: pd.DataFrame) -> EvaluationResult:
        grouped_uris = []
        grouped_scores = []
        targets = df.ext.gold_uris

        for group in df.ext.groupby_qid:
            scores = self._get_scores(group)
            grouped_scores.append(scores)
            grouped_uris.append(group.ext.uris)

        return compute_letor_scores(grouped_uris, grouped_scores, targets)

    def rank(self, group: pd.DataFrame) -> List[int]:
        scores = self._get_scores(group)
        idx = np.argsort(scores)
        return list(idx)

    def _get_scores(self, group: pd.DataFrame) -> List[float]:
        return list(range(len(group)))
