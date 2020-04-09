import operator
import os
import statistics
from collections import defaultdict

import more_itertools as mit
from rust_fst import Map
from sklearn.model_selection import KFold

import attr
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate

from tqdm import tqdm

from gleipnir.config import path_to_results
from gleipnir.corpora import load_aida_all
from gleipnir.datasets import get_raw_corpus_data
from gleipnir.formats import Corpus, Sentence, Document, List, Dict


@attr.s
class Score:
    precision: float = attr.ib()
    recall: float = attr.ib()
    f1: float = attr.ib()


class StringRecommender:

    def fit_evaluate(self, documents_train: List[Document], documents_eval: List[Document]) -> Score:
        model = build_frequency_dict(documents_train)

        gold = {}
        predictions = defaultdict(set)

        i = 0
        for doc in tqdm(documents_eval):
            for sentence in doc.sentences:
                if not len(sentence.entities):
                    continue

                for entity in sentence.entities.values():
                    key = (doc.name, sentence.idx, entity.start, entity.end)
                    gold[key] = entity.uri

                for n in [1, 2, 3]:
                    for (begin, end, mention) in generate_ngrams(sentence, n):
                        if mention not in model:
                            continue

                        top3_predictions = sorted(model[mention].items(), key=operator.itemgetter(1))[:3]
                        for prediction in top3_predictions:
                            key = (doc.name, sentence.idx, begin, end)
                            predictions[key].add(prediction[0])

            i += 1
            # if i > 10: break

        return precision_recall_f1(gold, predictions)

    def name(self) -> str:
        return f"String"

class LevenshteinRecommender:

    def __init__(self, n: int):
        self.n = n

    def fit_evaluate(self, documents_train: List[Document], documents_eval: List[Document]) -> Score:
        model = build_frequency_dict(documents_train)

        mentions = []
        labels = []

        # Just use the entity that was most often linked with this mention
        for mention, candidates in model.items():
            if candidates:
                label = max(candidates, key=candidates.get)
            else:
                label = ""

            mentions.append(mention)
            labels.append(label)

        le = LabelEncoder()
        le.fit(labels)
        items = [(k, v) for k, v in sorted(zip(mentions, le.transform(labels)))]

        m = Map.from_iter(items)

        gold = {}
        predictions = defaultdict(set)

        # Predict
        for doc in tqdm(documents_eval):
            for sentence in doc.sentences:
                if not len(sentence.entities):
                    continue

                for entity in sentence.entities.values():
                    key = (doc.name, sentence.idx, entity.start, entity.end)
                    gold[key] = entity.uri

                for n in [1, 2, 3]:
                    for (begin, end, mention) in generate_ngrams(sentence, n):
                        if len(mention) <= 3:
                            continue

                        for match, label_id in m.search(term=mention, max_dist=self.n):
                            # Only consider matches that have the same tokenization (heuristic)
                            if len(match) <= 3 or match.count(" ") != mention.count(" "):
                                continue

                            label = le.inverse_transform([label_id])[0]
                            key = (doc.name, sentence.idx, begin, end)
                            predictions[key].add(label)

        return precision_recall_f1(gold, predictions)

    def name(self) -> str:
        return f"Leven@{self.n}"


def build_frequency_dict(documents: List[Document]) -> Dict[str, Dict[str, int]]:
    model = defaultdict(lambda: defaultdict(int))

    for doc in documents:
        for sentence in doc.sentences:
            for entity in sentence.entities.values():
                mention = sentence.get_covered_text(entity).lower()

                model[mention][entity.uri] += 1

    return model


def generate_ngrams(sentence: Sentence, n: int):
    # We generate token n-grams
    for i, tokens in enumerate(mit.windowed(sentence.tokens, n)):
        if any(t is None for t in tokens):
            continue

        begin = i
        end = i + n
        text = " ".join(t.text for t in tokens)

        yield (begin, end, text.lower())


def precision_recall_f1(gold, predictions):
    tp = 0
    fp = 0
    fn = 0

    for span, gold_label in gold.items():
        if gold_label in predictions[span]:
            tp += 1
        else:
            fn += 1

    for span, predicted_labels in predictions.items():
        if span not in gold or gold[span] not in predicted_labels:
            fp += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    print("#Gold:", len(gold))
    print("#Predictions:", len(predictions))
    print("Precision: ", precision)
    print("Recall: ", recall)

    return Score(precision=precision, recall=recall, f1=f1)


def evaluate_on_train_test(model, corpus_train: Corpus, corpus_eval: Corpus):
    return model.fit_evaluate(corpus_train.documents, corpus_eval.documents)


def evaluate_with_cross_validation(model, corpus_all: Corpus):
    kf = KFold(n_splits=10)

    indices = range(corpus_all.number_of_documents())
    scores = []
    for idx_train, idx_eval in kf.split(indices):
        docs_train = [corpus_all.documents[i] for i in idx_train]
        docs_eval = [corpus_all.documents[i] for i in idx_eval]

        try:
            score = model.fit_evaluate(docs_train, docs_eval)
        except:
            continue

        scores.append(score)

    average_scores = Score(
        precision=statistics.mean(s.precision for s in scores),
        recall=statistics.mean(s.recall for s in scores),
        f1=statistics.mean(s.f1 for s in scores)
    )

    return average_scores


def main():
    names_train_test_split = [
        "aida",
        "wwo-fuseki",
        "1641-fuseki",
    ]

    names_cv = [
        "wwo-fuseki",
        "1641-fuseki",
    ]

    models = [
        StringRecommender(),
        LevenshteinRecommender(1),
        LevenshteinRecommender(2),
    ]

    scores = []

    # Train test
    for name in names_train_test_split:
        data = get_raw_corpus_data(name)

        print("Evaluating train test recommender for", name)

        for model in models:
            score = evaluate_on_train_test(model, data.corpus_train, data.corpus_dev)
            scores.append((name, model.name(), score))

    # Cross validation
    for name in names_cv:
        data = get_raw_corpus_data(name)

        print("Evaluating cv recommender for", name)

        for model in models:
            score = evaluate_with_cross_validation(model, data.corpus_all)
            scores.append((name, model.name(), score))

    for e in scores:
        print(e)

    headers = ["Dataset", "Model", "Precision", "Recall", "F1"]
    table = [(score[0], score[1], score[2].precision, score[2].recall, score[2].f1) for score in scores]

    results_dir = path_to_results()
    with open(os.path.join(results_dir, "scores_full.csv"), "w") as f:
        table_latex = tabulate(table, headers=headers, floatfmt=".2f", tablefmt="latex_booktabs")
        f.write(table_latex)

if __name__ == '__main__':
    main()