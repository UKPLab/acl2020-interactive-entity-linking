from typing import List, Dict, Any

import pandas as pd

from gleipnir.evaluation.metrics import EvaluationResult


class LetorModel:

    def __init__(self):
        self.model = None

    def fit_evaluate(self, df_train: pd.DataFrame, df_val: pd.DataFrame, verbose=False, **kwargs) -> EvaluationResult:
        raise NotImplementedError()

    def evaluate(self, df: pd.DataFrame) -> EvaluationResult:
        raise NotImplementedError()

    def rank(self, group: pd.DataFrame) -> List[int]:
        raise NotImplementedError()

    def explain(self, features: List[str]) -> List:
        return []









