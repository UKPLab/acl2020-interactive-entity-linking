from gleipnir.config import *
from gleipnir.formats import Corpus, read_conllel

# AIDA


def load_aida_train(with_labels: bool = True) -> Corpus:
    corpus = read_conllel(PATH_AIDA_WIKIDATA_TRAIN, with_labels)
    corpus.name = "aida_train"
    return corpus


def load_aida_dev(with_labels: bool = True) -> Corpus:
    corpus = read_conllel(PATH_AIDA_WIKIDATA_DEV, with_labels)
    corpus.name = "aida_dev"
    return corpus


def load_aida_test(with_labels: bool = True) -> Corpus:
    corpus = read_conllel(PATH_AIDA_WIKIDATA_TEST, with_labels)
    corpus.name = "aida_test"
    return corpus


def load_aida_all(with_labels: bool = True) -> Corpus:
    documents = []
    for c in (load_aida_train(with_labels), load_aida_dev(with_labels), load_aida_test(with_labels)):
        documents.extend(c.documents)
    corpus = Corpus(documents)
    corpus.name = "aida_all"
    return corpus

# WWO


def load_wwo_train(with_labels: bool = True) -> Corpus:
    corpus = read_conllel(PATH_WWO_TRAIN, with_labels)
    corpus.name = "wwo_train"
    return corpus


def load_wwo_dev(with_labels: bool = True) -> Corpus:
    corpus = read_conllel(PATH_WWO_DEV, with_labels)
    corpus.name = "wwo_dev"
    return corpus


def load_wwo_test(with_labels: bool = True) -> Corpus:
    corpus = read_conllel(PATH_WWO_TEST, with_labels)
    corpus.name = "wwo_test"
    return corpus


def load_wwo_all(with_labels: bool = True) -> Corpus:
    documents = []
    for c in (load_wwo_train(with_labels), load_wwo_dev(with_labels), load_wwo_test(with_labels)):
        documents.extend(c.documents)
    corpus = Corpus(documents)
    corpus.name = "wwo_all"
    return corpus


# Depositions


def load_depositions_train_raw(with_labels: bool = True) -> Corpus:
    corpus = read_conllel(PATH_DEPOSITIONS_TRAIN, with_labels)
    corpus.name = "1641_train"
    return corpus


def load_depositions_dev_raw(with_labels: bool = True) -> Corpus:
    corpus = read_conllel(PATH_DEPOSITIONS_DEV, with_labels)
    corpus.name = "1641_dev"
    return corpus


def load_depositions_test_raw(with_labels: bool = True) -> Corpus:
    corpus = read_conllel(PATH_DEPOSITIONS_TEST, with_labels)
    corpus.name = "1641_test"
    return corpus


def load_depositions_raw_all(with_labels: bool = True) -> Corpus:
    documents = []
    for c in (load_depositions_train_raw(with_labels), load_depositions_dev_raw(with_labels), load_depositions_test_raw(with_labels)):
        documents.extend(c.documents)
    corpus = Corpus(documents)
    corpus.name = "1641_all"
    return corpus


def load_depositions_train(with_labels: bool = True) -> Corpus:
    corpus = read_conllel(PATH_DEPOSITIONS_TRAIN_REMAPPED, with_labels)
    corpus.name = "1641_train"
    return corpus


def load_depositions_dev(with_labels: bool = True) -> Corpus:
    corpus = read_conllel(PATH_DEPOSITIONS_DEV_REMAPPED, with_labels)
    corpus.name = "1641_dev"
    return corpus


def load_depositions_test(with_labels: bool = True) -> Corpus:
    corpus = read_conllel(PATH_DEPOSITIONS_TEST_REMAPPED, with_labels)
    corpus.name = "1641_test"
    return corpus


def load_depositions_all(with_labels: bool = True) -> Corpus:
    documents = []
    for c in (load_depositions_train(with_labels), load_depositions_dev(with_labels), load_depositions_test(with_labels)):
        documents.extend(c.documents)
    corpus = Corpus(documents)
    corpus.name = "1641_all"
    return corpus


