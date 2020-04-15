import os
import statistics
from typing import List, Dict

import  attr
import pandas as pd

from tabulate import tabulate

from sklearn.model_selection import KFold

from gleipnir.config import path_to_results
from gleipnir.datasets import load_handcrafted_data, load_dataframe_from_csv
from gleipnir.evaluation.metrics import EvaluationResult
from gleipnir.models.bayesian import GpplLetorModel
from gleipnir.models.ranknet import HandcraftedRankNetLetorModel, RankNetWithEmbeddingsLetorModel
from gleipnir.models.ranksvm import SklearnRankSvmModel
from gleipnir.models.lgbm import LgbmModel
from gleipnir.models.baselines import DictionaryBaseline, LevenshteinBaseline


@attr.s
class Result:
    model_name: str = attr.ib()
    dataset_name: str = attr.ib()
    scores: EvaluationResult = attr.ib()
    explanation: List[Dict] = attr.ib(default=None)


def evaluate_lgbm(dataset_name: str, df_train, df_eval) -> Result:
    model = LgbmModel()

    result = model.fit_evaluate(df_train, df_eval)

    explanations = model.explain(df_train.ext.features)
    for explanation in explanations:
        explanation["dataset"] = dataset_name
        explanation["model"] = "lgbm"

    return Result("LGBM", dataset_name, result, explanations)


def evaluate_ranksvm(dataset_name: str, df_train, df_eval) -> Result:
    model = SklearnRankSvmModel()

    result = model.fit_evaluate(df_train, df_eval)
    explanations = model.explain(df_train.ext.features)
    for explanation in explanations:
        explanation["dataset"] = dataset_name
        explanation["model"] = "ranksvm"

    return Result("RankSVM", dataset_name, result, explanations)


def evaluate_ranknet_handcrafted(dataset_name: str, df_train, df_eval) -> Result:
    model = HandcraftedRankNetLetorModel(df_train.ext.num_features)
    result = model.fit_evaluate(df_train, df_eval)

    return Result("RankNet Handcrafted", dataset_name, result)


def evaluate_ranknet_embedded(dataset_name: str, df_train, df_eval) -> Result:
    model = RankNetWithEmbeddingsLetorModel()
    result = model.fit_evaluate(df_train, df_eval)

    return Result("RankNet Embeddings", dataset_name, result)


def evaluate_gppl(dataset_name: str, df_train, df_eval) -> Result:
    df_train = df_train.ext.subsample(100)
    # df_dev = df_dev.ext.subsample(10)

    model = GpplLetorModel()
    result = model.fit_evaluate(df_train, df_eval)

    return Result("GPPl", dataset_name, result)


def evaluate_dictionary_baseline(dataset_name: str, df_train, df_eval) -> Result:
    model = DictionaryBaseline()

    result = model.fit_evaluate(df_train, df_eval)

    return Result("MLEB", dataset_name, result)


def evaluate_leven_baseline(dataset_name: str, df_train, df_eval) -> Result:
    model = LevenshteinBaseline()

    result = model.fit_evaluate(df_train, df_eval)

    return Result("LSB", dataset_name, result)


def main():
    datasets_train_test = [
        # "aida",
        # "wwo-fuseki",
        # "1641-fuseki",
    ]

    datasets_cross_validation = [
        "wwo-fuseki",
        "1641-fuseki",
    ]

    eval_functions = [
        evaluate_dictionary_baseline,
        evaluate_leven_baseline,
        evaluate_lgbm,
        evaluate_ranksvm,
        evaluate_ranknet_handcrafted,
    ]

    evaluate_on_test = True

    results = []
    for dataset in datasets_train_test:
        df_train, df_eval = load_handcrafted_data(dataset, evaluate_on_test)
        
        r = [eval_function(dataset, df_train, df_eval) for eval_function in eval_functions]
        print(r)
        results.extend(r)

    for dataset in datasets_cross_validation:
        print(dataset)
        ds_full = load_dataframe_from_csv(dataset + "_all")
        qids = ds_full["qid"]
        indices = qids.unique()

        cv_results = []

        for eval_function in eval_functions:
            kf = KFold(n_splits=10)
            for train, test in kf.split(indices):
                df_train = ds_full[qids.isin(train)]
                df_eval = ds_full[qids.isin(test)]

                cv_result = eval_function(dataset, df_train, df_eval)
                cv_results.append(cv_result)

            average_cv_result = EvaluationResult(
                accuracy = statistics.mean(r.scores.accuracy for r in cv_results),
                accuracy_top5 = statistics.mean(r.scores.accuracy_top5 for r in cv_results),
                mean_reciprocal_rank = statistics.mean(r.scores.mean_reciprocal_rank for r in cv_results),
                mean_number_of_candidates = statistics.mean(r.scores.mean_number_of_candidates for r in cv_results),
                gold_not_in_candidates = statistics.mean(r.scores.gold_not_in_candidates for r in cv_results),
            )

            results.append(Result(cv_result.model_name, cv_result.dataset_name, average_cv_result))

    headers = ["Dataset", "Model",  "Acc@1", "Acc@5", "MAP", "C"]
    table = [
        (r.dataset_name, r.model_name, r.scores.accuracy, r.scores.accuracy_top5, r.scores.mean_reciprocal_rank, r.scores.mean_number_of_candidates)
        for r in results
    ]
    print("\n" + tabulate(table, headers=headers, floatfmt=".2f"))

    all_results = []
    for r in results:
        if r.explanation:
            all_results.extend(r.explanation)

    df = pd.DataFrame(all_results)

    results_dir = path_to_results()
    with open(os.path.join(results_dir, "scores_full.csv"), "w") as f:
        table_latex = tabulate(table, headers=headers, floatfmt=".2f", tablefmt="latex_booktabs")
        f.write(table_latex)

    df.to_csv(os.path.join(results_dir, f"explanations.csv"), index=False)


if __name__ == '__main__':
    #import cProfile
    #cProfile.run("main()", "restats")

    main()

