import logging
from typing import List

import numpy as np
import pandas as pd

from gleipnir.evaluation.metrics import EvaluationResult, compute_letor_scores
from gleipnir.models.gppl.gp_pref_learning import GPPrefLearning
from gleipnir.models.letor_models import LetorModel
from gleipnir.util import get_logger

logger = get_logger(__name__)


class GpplLetorModel(LetorModel):

    def fit_evaluate(self, df_train: pd.DataFrame, df_val: pd.DataFrame, verbose=False, features = None, **kwargs) -> EvaluationResult:
        positive_preference_ids = []
        negative_preference_ids = []
        labels = []

        item_id = 0

        for group in df_train.ext.groupby_qid:
            positive_id = item_id
            item_id += 1

            X_g = group.ext.X
            y_g = group.ext.y

            yp = y_g[0]
            y_n = y_g[1:]

            assert yp == 1.0
            assert all(yn == 0.0 for yn in y_n)

            for _ in range(len(group) - 1):
                positive_preference_ids.append(positive_id)
                negative_preference_ids.append(item_id)
                labels.append(1)
                item_id += 1

        positive_preference_ids = np.array(positive_preference_ids)
        negative_preference_ids = np.array(negative_preference_ids)
        labels = np.array(labels)
        features = df_train.ext.X

        assert len(positive_preference_ids) == len(negative_preference_ids) == len(labels)
        assert positive_preference_ids.max() < features.shape[0]
        assert negative_preference_ids.max() < features.shape[0]

        logger.info("Number of preferences: [%s]", len(positive_preference_ids))
        logger.info("Feature size: [%s]", {features.shape})

        logger.info("Starting training")
        model = GPPrefLearning(df_train.ext.X.shape[1], shape_s0=2, rate_s0=200)
        model.fit(positive_preference_ids, negative_preference_ids, features, labels, optimize=False)

        logger.info("Training done")
        self.model = model

        return self.evaluate(df_val)

    def evaluate(self, df: pd.DataFrame, verbose=False) -> EvaluationResult:
        logger.info("Evaluating")
        grouped_scores = []
        grouped_uris = []
        targets = df.ext.gold_uris

        results = pd.DataFrame(
            {
                "scores": self.model.predict_f(df.ext.X)[0].squeeze(),
                "qid": df["qid"].values
            }
        )

        for group, result in zip(df.ext.groupby_qid, results.ext.groupby_qid):
            grouped_uris.append(group.ext.uris)
            grouped_scores.append(result["scores"].values)

        return compute_letor_scores(grouped_uris, grouped_scores, targets)

    def rank(self, *input) -> List[str]:
        pass
