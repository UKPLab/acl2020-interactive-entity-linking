# See e.g. 'Evaluating Entity Linking: An Analysis of Current Benchmark Datasets and a Roadmap for Doing a Better Job'
# http://www.lrec-conf.org/proceedings/lrec2016/pdf/926_Paper.pdf
# https://github.com/dbpedia-spotlight/evaluation-datasets
import statistics

import attr
from collections import defaultdict
from typing import Dict, List

from jellyfish import levenshtein_distance
import numpy as np
import textdistance
from tabulate import tabulate
from tqdm import tqdm

from gleipnir.corpora import *
from gleipnir.kb import *



@attr.s
class CorpusStatistics:
    name = attr.ib()                        # type: str
    number_of_documents = attr.ib()         # type: int
    number_of_tokens = attr.ib()            # type: int
    number_of_entities = attr.ib()          # type: int
    number_of_nil = attr.ib()               # type: int
    entities_per_sentence = attr.ib()       # type: float
    confusability_corpus = attr.ib()        # type: Score
    confusability_kb = attr.ib()            # type: Score
    number_of_no_candidates = attr.ib()     # type: int
    gini = attr.ib()                        # type: float
    distance_mention_label = attr.ib()      # type: float


@attr.s
class Score:
    average = attr.ib()                     # type: float
    min = attr.ib()                         # type: int
    max = attr.ib()                         # type: int
    std = attr.ib()                         # type: float

    @staticmethod
    def from_scores(scores: List[int]) -> 'Score':
        arr = np.array(scores)
        return Score(
            average=np.average(arr),
            min=np.min(arr),
            max=np.max(arr),
            std=np.std(scores)
        )

def generate_statistics(corpora: List[Corpus], kb: KnowledgeBase) -> List[CorpusStatistics]:
    result = []

    for corpus in corpora:
        stats = _generate_base_statistics(corpus)

        # Confusability
        text_to_relation_index = defaultdict(list)
        kb_confusability_scores = []
        number_of_nil = 0

        number_of_no_candidates = 0

        entity_count = defaultdict(int)

        distances_mention_label = []

        for document in tqdm(corpus.documents):
            for sentence in document.sentences:
                for entity in sentence.entities.values():
                    surface_form = sentence.get_covered_text(entity).lower()

                    text_to_relation_index[surface_form].append(entity.uri)

                    score = len(kb.search_mention(surface_form.lower()))
                    kb_confusability_scores.append(score)
                    entity_count[entity.uri] += 1

                    if score == 0:
                        number_of_no_candidates += 1

                    if entity.uri == "NIL":
                        number_of_nil += 1

                    gold = kb.get_by_iri(entity.uri)
                    distances_mention_label.append(
                        levenshtein_distance(gold.label.lower(), surface_form.lower())
                    )

        stats.confusability_corpus = Score.from_scores([len(set(x)) for x in text_to_relation_index.values()])
        stats.confusability_kb = Score.from_scores(kb_confusability_scores)
        stats.dominance = _compute_dominance_scores(text_to_relation_index)
        stats.number_of_nil = number_of_nil / corpus.number_of_entities() * 100
        stats.number_of_no_candidates = number_of_no_candidates
        stats.gini = gini(np.array(list(entity_count.values())))
        stats.distance_mention_label = statistics.mean(distances_mention_label)

        result.append(stats)

    return result

def generate_statistics_wwo() -> List[CorpusStatistics]:
    # corpora = [load_wwo_train(), load_wwo_dev(), load_wwo_test()]
    corpora = [load_wwo_all()]
    kb = FusekiKnowledgeBase("wwo")

    return generate_statistics(corpora, kb)


def generate_statistics_depositions() -> List[CorpusStatistics]:
    # corpora = [load_depositions_train(), load_depositions_dev(), load_depositions_test()]
    corpora = [load_depositions_all()]
    kb = FusekiKnowledgeBase("depositions")

    return generate_statistics(corpora, kb)



def generate_statistics_aida() -> List[CorpusStatistics]:
    corpora = [load_aida_all()]
    kb = WikidataVirtuosoKnowledgeBase()

    return generate_statistics(corpora, kb)


def _generate_base_statistics(corpus: Corpus) -> CorpusStatistics:
    return CorpusStatistics(
        name=corpus.name,
        number_of_documents=corpus.number_of_documents(),
        number_of_tokens=corpus.number_of_tokens(),
        number_of_entities=corpus.number_of_entities(),
        number_of_nil=0,
        entities_per_sentence=corpus.entities_per_sentence(),
        confusability_corpus=None,
        confusability_kb=None,
        number_of_no_candidates=0,
        gini=0.0,
        distance_mention_label=0.0
    )


def _compute_dominance_scores(index: Dict[str, List[str]]) -> Score:
    """ surface -> list of relations that were linked to this surface form """
    counts = []
    totals = []

    for surface_form, entities in index.items():
        for entity in entities:
            c = entities.count(entity)
            counts.append(c)
            totals.append(max(len(entities) - c, c))

    counts_arr = np.array(counts, dtype=int)
    totals_arr = np.array(totals)

    dominance = counts_arr / totals_arr

    return Score(
        average=np.average(dominance),
        min=np.min(counts_arr),
        max=np.max(counts_arr),
        std=np.std(dominance)
    )


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # https://github.com/oliviaguest/gini/blob/master/gini.py
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten().astype(float)
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))


def main():
    fns = [
        generate_statistics_aida,
        generate_statistics_wwo,
        generate_statistics_depositions,
    ]

    stats = []
    for fn in fns:
        stats.extend(fn())

    headers = ["Corpus", "#D", "#T", "#E", "#E/S", "%NIL", "Avg. Amb.", "Conf.", "Gini"]
    rows = []
    for e in stats:
        row = [
            e.name,
            e.number_of_documents,
            e.number_of_tokens,
            e.number_of_entities,
            e.entities_per_sentence,
            e.number_of_nil,
            e.confusability_corpus.average,
            e.confusability_kb.average,
            e.gini,
        ]
        rows.append(row)

    print("\n" + tabulate(rows, headers=headers, floatfmt=".2f", tablefmt="latex_booktabs"))


if __name__ == '__main__':
    main()
