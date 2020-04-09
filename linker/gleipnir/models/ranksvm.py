from typing import List, Dict, Any

import numpy as np
import pandas as pd

from scipy import linalg

from sklearn import svm
from sklearn.preprocessing import normalize

from gleipnir.evaluation.metrics import EvaluationResult, compute_letor_scores
from gleipnir.models.letor_models import LetorModel


class SklearnRankSvmModel(LetorModel):

    def __init__(self):
        super().__init__()

    def fit(self, df_train: pd.DataFrame, verbose=False):
        X_train = df_train.ext.X
        y_train = df_train.ext.y
        groups_train = df_train.ext.group_sizes

        assert len(X_train) == len(y_train) == sum(groups_train), f"Sizes are not equal - Xt: {len(X_train)}, yt {len(y_train)}, Gt:{ sum(groups_train)}"

        X_train, y_train = self._pairwise_transform(df_train)
        clf = svm.LinearSVC(dual=False, verbose=verbose)
        clf.fit(X_train, y_train)

        self.model = clf
        self._weights = self.model.coef_.ravel() / linalg.norm(self.model.coef_)

    def fit_evaluate(self, df_train: pd.DataFrame, df_val: pd.DataFrame, verbose=False, **kwargs) -> EvaluationResult:
        self.fit(df_train, verbose)
        return self.evaluate(df_val)

    def evaluate(self, df: pd.DataFrame) -> EvaluationResult:
        grouped_uris = []
        grouped_scores = []
        targets = df.ext.gold_uris

        for group in df.ext.groupby_qid:
            scores = group.ext.X.dot(self._weights)

            grouped_scores.append(scores)
            grouped_uris.append(group.ext.uris)

        return compute_letor_scores(grouped_uris, grouped_scores, targets)

    def _pairwise_transform(self, df: pd.DataFrame):
        print("Converting")

        X_all = []
        y_all = []
        gold_indices = df.ext.gold_indices

        for group_idx, group in enumerate(df.ext.groupby_qid):
            X_g = group.ext.X
            y_g = group.ext.y
            gold_idx = gold_indices[group_idx]

            assert gold_idx >= 0, "Group does not have gold label!"
            assert y_g[gold_idx] == 1.0, "Gold should have score of 1!"

            x_p = X_g[gold_idx]
            y_p = y_g[gold_idx]

            X_n = np.delete(X_g, gold_idx, axis=0)
            y_n = np.delete(y_g, gold_idx, axis=0)

            assert len(X_n) == len(y_n)
            k = len(y_n)

            X_all.append(x_p - X_n)
            y_all.append(np.ones(k))

            X_all.append(X_n - x_p)
            y_all.append(- np.ones(k))

        return np.concatenate(X_all), np.concatenate(y_all)

    def rank(self, group: pd.DataFrame) -> List[int]:
        scores = group.ext.X.dot(self._weights)

        scored_group = group.assign(rerank_scores=scores)
        scored_group.sort_values("rerank_scores", inplace=True, ascending=False)

        result = scored_group["candidate_id"].astype(np.int).values

        return list(result)

    def explain(self, features: List[str]) -> List:
        # Gene Selection for Cancer Classification using Support Vector Machines

        results = []
        scores = self._weights * self._weights
        scores = normalize(scores.reshape(1, -1), norm="l1").squeeze()

        for w, l in zip(scores, features):
            results.append({"feature": l, "weight": w})

        return results
