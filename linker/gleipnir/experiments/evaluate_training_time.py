import os
import statistics
from timeit import default_timer as timer

from tabulate import tabulate
from tqdm import tqdm

from gleipnir.config import path_to_results
from gleipnir.datasets import load_dataframe_from_csv
from gleipnir.experiments.evaluate_ranking import evaluate_dictionary_baseline, evaluate_leven_baseline, evaluate_lgbm, \
    evaluate_ranksvm, evaluate_ranknet_handcrafted


def main():
    datasets = [
        "aida",
        "wwo-fuseki",
        "1641-fuseki",
    ]

    eval_functions = [
        evaluate_lgbm,
        evaluate_ranksvm,
        evaluate_ranknet_handcrafted,
    ]

    repetitions = 10

    rows = []
    for name in tqdm(datasets):
        print("Measuring time for ", name)
        ds_train_name = name + "_all"
        ds_eval_name = name + "_dev"

        df_train = load_dataframe_from_csv(ds_train_name)
        df_eval = load_dataframe_from_csv(ds_eval_name)

        for eval_function in eval_functions:
            times = []
            for _ in range(repetitions):
                start = timer()
                score = eval_function(name, df_train, df_eval)
                end = timer()

                times.append(end - start)

            row = (score.dataset_name, score.model_name, statistics.mean(times))
            rows.append(row)

    headers = [
        "Model",
        "Dataset",
        "t"
    ]
    results_dir = path_to_results()
    with open(os.path.join(results_dir, "runtimes.csv"), "w") as f:
        table_latex = tabulate(rows, headers=headers, floatfmt=".2f", tablefmt="latex_booktabs")
        f.write(table_latex)


if __name__ == '__main__':
    main()