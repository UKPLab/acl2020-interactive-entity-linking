from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


import lightgbm as lgb

from gleipnir.evaluation.metrics import EvaluationResult, compute_letor_scores
from gleipnir.models.letor_models import LetorModel


class LgbmModel(LetorModel):

    def __init__(self):
        super().__init__()

    def fit_evaluate(self, df_train: pd.DataFrame, df_val: pd.DataFrame, verbose=False, features = None, **kwargs) -> EvaluationResult:
        X_train = df_train.ext.X
        y_train = df_train.ext.y
        groups_train = df_train.ext.group_sizes
        X_val = df_val.ext.X
        y_val = df_val.ext.y
        groups_val = df_val.ext.group_sizes

        assert len(X_train) == len(y_train) == sum(groups_train), f"Sizes are not equal - Xt: {len(X_train)}, yt {len(y_train)}, Gt:{ sum(groups_train)}"
        assert len(X_val) == len(y_val) == sum(groups_val), f"Sizes are not equal - Xv: {len(X_val)}, yv {len(y_val)}, Gv:{sum(groups_val)}"

        gbm = lgb.LGBMRanker(boosting_type="gbdt", class_weight="balanced", n_estimators=200)
        gbm.fit(X_train, y_train, group=groups_train,
                eval_set=[(X_val, y_val)],
                eval_group=[groups_val],
                early_stopping_rounds=10,
                feature_name=features, verbose=verbose)

        self.model = gbm

        return self.evaluate(df_val)

    def evaluate(self, df: pd.DataFrame, verbose=False) -> EvaluationResult:
        grouped_uris = []
        grouped_scores = []
        targets = df.ext.gold_uris

        for group in df.ext.groupby_qid:
            scores = self.model.predict(group.ext.X, num_iteration=self.model.best_iteration_)

            grouped_uris.append(group.ext.uris)
            grouped_scores.append(scores)

        return compute_letor_scores(grouped_uris, grouped_scores, targets)

    def rank(self, group: pd.DataFrame) -> List[int]:
        scores = self.model.predict(group.ext.X, num_iteration=self.model.best_iteration_)
        idx = np.argsort(scores)[::-1]
        return list(idx)

    def explain(self, features: List[Dict]) -> List:
        # Gene Selection for Cancer Classification using Support Vector Machines

        results = []
        scores = normalize(self.model.booster_.feature_importance(importance_type="split").reshape(1, -1), norm="l1").squeeze()
        for w, l in zip(scores, features):
            results.append({"feature": l, "weight": w})

        return results
