import os
from typing import Dict

import pandas as pd

import matplotlib.pyplot as plt
import pylab

from gleipnir.config import path_to_results
from gleipnir.datasets import load_handcrafted_simulation_data
from gleipnir.evaluation.metrics import EvaluationResult
from gleipnir.models.baselines import DictionaryBaseline, LevenshteinBaseline
from gleipnir.models.bayesian import GpplLetorModel
from gleipnir.models.letor_models import LetorModel
from gleipnir.models.lgbm import LgbmModel
from gleipnir.models.ranknet import HandcraftedRankNetLetorModel
from gleipnir.models.ranksvm import SklearnRankSvmModel
from gleipnir.util import get_logger

logger = get_logger(__name__)


class SliceSizeGenerator:

    def __init__(self):
        self.n = 10

    def __iter__(self):
        self.n = 10
        return self

    def __next__(self):
        x = self.n

        if x < 100:
            self.n += 10
        elif x < 1000:
            self.n += 50
        elif x < 10000:
            self.n += 200
        else:
            self.n += 500

        return x


def annotate_all(df: pd.DataFrame) -> pd.DataFrame:
    frames = []

    for group in df.ext.groupby_qid:
        annotations = group

        # We need to have at least gold and a negative example
        if len(annotations) >= 2:
            frames.append(annotations)

    return pd.concat(frames)


def annotate_above_gold(model: LetorModel, df: pd.DataFrame) -> pd.DataFrame:
    frames = []

    annotation_rations = []

    for group in df.ext.groupby_qid:
        ranks = model.rank(group)
        ranked_group = group.assign(rank=ranks)
        ranked_group.sort_values("rank", inplace=True)
        ranked_group.drop('rank', axis=1, inplace=True)

        # We map the gold index from the unranked group to the ranked group
        gold_idx = ranks[group.ext.gold_indices[0]]
        assert ranked_group["score"].values[gold_idx] == 1.0

        # As an example, we have a list of candidates
        # 1. Candidate one
        # 2. Candidate two
        # 3. Candidate three (Gold)
        # 4. Candidate four
        # We then have 1 and 2 as negative candidates and 3 as the gold positive one
        # In case there is no gold, we take all candidates as negative and add the gold as positive anyways.
        # This is similar to the user finding the right candidate by submitting search queries to the KB
        if gold_idx >= 0:
            annotations = ranked_group[:gold_idx + 1].copy()
            annotations["gold_idx"] = gold_idx
            assert annotations["score"].values[gold_idx] == 1.0
        else:
            annotations = group

        # We need to have at least gold and a negative example
        if len(annotations) >= 2:
            frames.append(annotations)

        annotation_rations.append(len(annotations) / len(group))

    logger.info("Ratio of #preferences / # candidates: [%f]", sum(annotation_rations) / len(annotation_rations))

    return pd.concat(frames)


def plot_simulation(dir: str, data: Dict[int, EvaluationResult], model_name: str, dataset_name: str, split_name: str):
    t = []
    accuracy = []
    accuracy_top5 = []
    mrr = []

    for number_of_annotations in sorted(data.keys()):
        score = data[number_of_annotations]
        t.append(number_of_annotations)
        accuracy.append(score.accuracy)
        accuracy_top5.append(score.accuracy_top5)
        mrr.append(score.mean_reciprocal_rank)

    plt.clf()
    pylab.plot(t, accuracy,  'r', label="Accuracy@1")
    pylab.plot(t, accuracy_top5, 'b', label="Accuracy@5")
    pylab.plot(t, mrr, 'g', label="MRR")
    pylab.xlabel("Number of annotations")
    pylab.ylim(0.0, 1.0)
    pylab.legend(loc='lower right')

    title = f"{model_name} - {dataset_name} - {split_name}"
    plt.title(title)
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, f"{dataset_name}_{model_name}_{split_name}.pdf"))

    # plt.show()


def simulate_single(df: pd.DataFrame, model: LetorModel, seed_size: int):
    seed, df_train = df.ext.split_by_qid(seed_size)
    annotated_so_far = seed

    results = {}

    # Generate steps, make sure to train on all at the end
    number_of_groups = len(df_train.ext.groupby_qid)
    steps = []
    for slice_size in SliceSizeGenerator():
        if slice_size < number_of_groups:
            steps.append(slice_size)
        else:
            steps.append(number_of_groups)
            break

    last_group_size = 0
    for step_size in steps:
        logger.info(f"Running on annotations [%d..%d]", last_group_size, step_size)
        result = model.fit_evaluate(annotated_so_far, df)
        logger.debug(result)

        train_slice = df_train.ext.slice_by_qid(last_group_size, step_size)
        if train_slice.empty:
            break

        annotated_slice = annotate_all(train_slice)

        annotated_so_far = pd.concat([annotated_so_far, annotated_slice])

        results[step_size] = result
        last_group_size = step_size

    return results


def get_model(model_name: str, number_of_features: int) -> LetorModel:
    if model_name == "lgbm":
        return LgbmModel()
    elif model_name == "ranksvm":
        return SklearnRankSvmModel()
    elif model_name == "ranknet_handcrafted":
        return HandcraftedRankNetLetorModel(number_of_features)
    elif model_name == "gppl":
        return GpplLetorModel()
    elif model_name == "dictionary_baseline":
        return DictionaryBaseline()
    elif model_name == "levenshtein_baseline":
        return LevenshteinBaseline()
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def main():
    SEED_SIZE = 10

    model_names = [
        "lgbm",
        #"ranknet_handcrafted",
        "ranksvm",
        "dictionary_baseline",
        "levenshtein_baseline"
    ]

    dataset_names = [
        "wwo-fuseki",
        # "1641-fuseki",
        # "aida",
    ]

    params = {
        "n_epochs": 30
    }

    dir = path_to_results()

    all_results = []
    all_explanations = []

    for dataset_name in dataset_names:
        for model_name in model_names:
            logger.info("Running simulation for model [%s] and dataset [%s]", model_name, dataset_name)

            df = load_handcrafted_simulation_data(dataset_name)
            model = get_model(model_name, df.ext.num_features)

            results = simulate_single(df, model, SEED_SIZE)

            plot_simulation(dir, results, model_name, dataset_name, "All")

            for n, score in results.items():
                all_results.append({"n": n,
                                    "dataset": dataset_name,
                                    "model": model_name,
                                    "acc1": score.accuracy,
                                    "acc5": score.accuracy_top5,
                                    "map": score.mean_reciprocal_rank,
                                    "candidates": score.mean_number_of_candidates
                                    })

            explanations = model.explain(df.ext.features)
            if explanations:
                for explanation in explanations:
                    explanation["dataset"] = dataset_name
                    explanation["model"] = model_name
                all_explanations.extend(explanations)

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(dir, "scores.csv"), index=False)

    df = pd.DataFrame(all_explanations)
    df.to_csv(os.path.join(dir, f"explanations.csv"), index=False)


if __name__ == '__main__':
    main()
