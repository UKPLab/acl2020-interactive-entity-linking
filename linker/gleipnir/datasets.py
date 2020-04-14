import logging
import os
import random
from typing import List, Tuple

import attr

import numpy as np
import pandas as pd

from flair.data import Sentence

from sklearn.datasets import dump_svmlight_file
from torch.utils.data import Dataset

from gleipnir.corpora import *
from gleipnir.data import CandidateGenerator
from gleipnir.kb import FusekiKnowledgeBase, WikidataKnowledgeBase, KnowledgeBase
from gleipnir.util import get_logger

logger = get_logger(__name__)


class LetorDataset:
    pass

@attr.s
class TrainingData:
    corpus_name: str = attr.ib()
    corpus_train: Corpus = attr.ib()
    corpus_dev: Corpus = attr.ib()
    corpus_test: Corpus = attr.ib()
    corpus_all: Corpus = attr.ib()
    kb: KnowledgeBase = attr.ib()
    cg: CandidateGenerator = attr.ib()


@pd.api.extensions.register_dataframe_accessor("ext")
class HandcraftedExtensionAccessor:

    def __init__(self, pandas_obj: pd.DataFrame):
        self._df = pandas_obj
        self.name = ""

    @property
    def candidate_ids(self):
        return self._df["candidate_id"]

    @property
    def number_of_groups(self) -> int:
        return self._df["qid"].nunique()

    @property
    def X(self):
        return self._df[self.features].astype('float32').values

    @property
    def y(self):
        return self._df["score"].astype('float32').values

    @property
    def uris(self):
        return self._df["uri"].values

    @property
    def features(self) -> List[str]:
        return [f for f in self._df.columns if f.startswith("feat_")]

    @property
    def num_features(self) -> int:
        return len(self.features)

    @property
    def group_sizes(self) -> List[int]:
        return [int(x) for x in self._df["qid"].value_counts(sort=False)]

    @property
    def groupby_qid(self) -> List[pd.DataFrame]:
        return [_slice for (_, _slice) in self._df.groupby(["qid"])]

    @property
    def group_X(self) -> List[np.array]:
        return [_slice.ext.X for (_, _slice) in self._df.groupby(["qid"])]

    @property
    def group_y(self) -> List[np.array]:
        return [_slice.ext.y for (_, _slice) in self._df.groupby(["qid"])]

    @property
    def mentions(self) -> List[str]:
        return [_slice["mention"].iloc[0] for (_, _slice) in self._df.groupby(["qid"])]

    @property
    def labels(self) -> List[List[str]]:
        return [_slice["label"] for (_, _slice) in self._df.groupby(["qid"])]

    @property
    def contexts(self) -> List[str]:
        return [_slice["context"].iloc[0] for (_, _slice) in self._df.groupby(["qid"])]

    @property
    def gold_uris(self) -> List[str]:
        return [_slice["gold"].iloc[0] for (_, _slice) in self._df.groupby(["qid"])]

    @property
    def gold_indices(self) -> List[int]:
        return [_slice["gold_idx"].iloc[0] for (_, _slice) in self._df.groupby(["qid"])]

    def split_by_qid(self, qid) -> Tuple[pd.DataFrame, pd.DataFrame]:
        p1 = self._df.query(f"qid < {qid}")
        p2 = self._df.query(f"qid >= {qid}")
        return p1, p2

    def to_csv(self):
        assert self.name, "Need to set name when saving to csv"
        self._df.to_csv(os.path.join(PATH_HANDCRAFTED, f"{self.name}.csv"), index=False, sep="\t")

    def subsample(self, number_of_groups: int) -> pd.DataFrame:
        """ Selects the first `number_of_groups` groups. """
        # We assume that qids are sorted ascending
        limit = self._df["qid"].values[0] + number_of_groups
        result = self._df.query(f"qid < {limit}")
        assert len(result.ext.groupby_qid) == number_of_groups
        return result

    def slice_by_qid(self, lower: int, upper: int) -> pd.DataFrame:
        # We need to find the first qid as the offset
        offset = self._df["qid"].values[0]
        return self._df.query(f"qid >= {offset + lower} and qid < {offset + upper}")

    def to_svmlight(self):
        assert self.name, "Need to set name when saving to csv"
        dump_svmlight_file(self.X, self.y, os.path.join(PATH_HANDCRAFTED, f"{self.name}.dat"), query_id=self._df["qid"])


def get_raw_corpus_data(s: str, caching: bool = True):
    if s == "aida":
        data_train = load_aida_train()
        data_dev = load_aida_dev()
        data_test = load_aida_test()
        data_all = load_aida_all()

        kb = WikidataKnowledgeBase(caching=caching)
    elif s == "wwo-fuseki":
        data_train = load_wwo_train()
        data_dev = load_wwo_dev()
        data_test = load_wwo_test()
        data_all = load_wwo_all()

        kb = FusekiKnowledgeBase(name="wwo", caching=caching)
    elif s == "1641-fuseki":
        data_train = load_depositions_train()
        data_dev = load_depositions_dev()
        data_test = load_depositions_test()
        data_all = load_depositions_all()

        kb = FusekiKnowledgeBase(name="depositions", caching=caching)
    else:
        raise Exception(f"Unknown corpus name: {s}")

    cg = CandidateGenerator(kb)
    return TrainingData(s, data_train, data_dev, data_test, data_all, kb, cg)



def load_dataframe_from_csv(name: str) -> pd.DataFrame:
    p = os.path.join(PATH_HANDCRAFTED, f"{name}.csv")
    df = pd.read_csv(p, sep="\t")
    df.ext.name = name
    return df


def load_handcrafted_data(name: str, evaluate_on_test: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading [%s]", name)

    ds_train_name = name + "_train"
    ds_test_name = name + ("_dev" if not evaluate_on_test else "_test")

    df_train = load_dataframe_from_csv(ds_train_name)
    df_eval = load_dataframe_from_csv(ds_test_name)

    df_train.fillna('<unk>', inplace=True)
    df_eval.fillna('<unk>', inplace=True)

    return df_train, df_eval


def load_handcrafted_simulation_data(name: str) -> pd.DataFrame:
    logger.info("Loading [%s]", name)

    ds_name = f"{name}_full_sim"
    df = load_dataframe_from_csv(ds_name)

    df.fillna('<unk>', inplace=True)

    return df

class HandcraftedLetorDataset(Dataset):
    # https://github.com/yutayamazaki/RankNet-PyTorch/

    def __init__(self, df: pd.DataFrame):
        group_sizes = df.ext.group_sizes
        qids = df["qid"]
        indices = qids.unique()
        large_enough_groups = {i for i, group_size in zip(indices, group_sizes) if group_size >= 2}

        df = df[qids.isin(large_enough_groups)]

        self.X_grouped = df.ext.group_X
        self.y_grouped = df.ext.group_y
        self.gold_indices = df.ext.gold_indices

        assert len(self.X_grouped) == len(self.y_grouped) == len(self.gold_indices), "Groups have to have the same length!"

    def __len__(self) -> int:
        return len(self.y_grouped)

    def __getitem__(self, group_idx: int):
        X = self.X_grouped[group_idx]
        y = self.y_grouped[group_idx]
        gold_idx = self.gold_indices[group_idx]

        assert gold_idx >= 0, "Group does not have gold label!"
        assert y[gold_idx] == 1.0, "Gold should have score of 1!"

        x_p = X[gold_idx]
        y_p = y[gold_idx]

        indices = list(range(len(y)))
        indices.remove(gold_idx)
        idx_n = random.choice(indices)

        assert idx_n != gold_idx

        x_n = X[idx_n]
        y_n = y[idx_n]

        return {
           "x_p": x_p,
           "x_n": x_n,
           "y_p": y_p,
           "y_n": y_n
        }


class PairwiseFlairLetorDataset(Dataset):
    # https://github.com/yutayamazaki/RankNet-PyTorch/

    def __init__(self, df: pd.DataFrame):
        mentions = []
        grouped_kb_labels = []
        grouped_descriptions = []
        contexts = []

        for group in df.ext.groupby_qid:
            # The mention is identical for all items in the group
            mentions.append(Sentence(group["mention"].values[0], use_tokenizer=False))
            grouped_kb_labels.append([Sentence(x, use_tokenizer=True) for x in group["label"]])
            grouped_descriptions.append([Sentence(x, use_tokenizer=True) for x in group["description"]])
            contexts.append(Sentence(group["context"].values[0], use_tokenizer=True))

        self.mentions = mentions
        self.grouped_kb_labels = grouped_kb_labels
        self.grouped_descriptions = grouped_descriptions
        self.contexts = contexts
        self.y_grouped = df.ext.group_y
        self.gold_indices = df.ext.gold_indices

    def __len__(self) -> int:
        return len(self.y_grouped)

    def __getitem__(self, group_idx: int):
        mention = self.mentions[group_idx]
        labels = self.grouped_kb_labels[group_idx]
        descriptions = self.grouped_descriptions[group_idx]
        context = self.contexts[group_idx]
        y = self.y_grouped[group_idx]
        gold_idx = self.gold_indices[group_idx]

        assert gold_idx >= 0, "Group does not have gold label!"
        assert y[gold_idx] == 1.0, "Gold should have score of 1!"

        label_p = labels[gold_idx]
        description_p = descriptions[gold_idx]
        y_p = y[gold_idx]

        indices = np.arange(start=1, stop=len(y))
        idx_n = np.random.choice(indices)

        label_n = labels[idx_n]
        description_n = descriptions[idx_n]
        y_n = y[idx_n]

        return {
           "mention": mention,
           "label_p": label_p,
           "description_p": description_p,
           "y_p": y_p,
           "label_n": label_n,
           "description_n": description_n,
           "y_n": y_n,
           "context": context
        }
