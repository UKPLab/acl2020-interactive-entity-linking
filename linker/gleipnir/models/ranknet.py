from collections import defaultdict
from typing import List, Dict, Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from flair.data import Sentence
from flair.embeddings import WordEmbeddings, BytePairEmbeddings, CharacterEmbeddings, DocumentPoolEmbeddings, Embeddings
#
from more_itertools import split_into
from tqdm import tqdm, trange

from gleipnir.datasets import load_handcrafted_data, HandcraftedLetorDataset, PairwiseFlairLetorDataset
from gleipnir.models.letor_models import LetorModel
from gleipnir.evaluation.metrics import compute_letor_scores, EvaluationResult
from gleipnir.models.pytorchtools import EarlyStopping


class RankNet(nn.Module):
    def __init__(self, num_features: int, device):
        super(RankNet, self).__init__()

        self._device = device

        self.model = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.output_sig = nn.Sigmoid()

        self.to(device)

    def score_pair(self, batch: Dict[str, torch.Tensor]):
        x_p = batch["x_p"]
        x_n = batch["x_n"]

        s1 = self(x_p)
        s2 = self(x_n)

        out = self.output_sig(s1 - s2)

        return out.squeeze()

    def forward(self, batch: torch.Tensor):
        s = self.model(batch.to(self._device))
        return s.squeeze(1)

    @torch.no_grad()
    def evaluate(self, df: pd.DataFrame) -> EvaluationResult:
        grouped_scores = []
        grouped_uris = []
        targets = df.ext.gold_uris

        results = pd.DataFrame(
            {
                "scores": self.forward(torch.from_numpy(df.ext.X).float()).cpu().numpy(),
                "qid": df["qid"].values
            }
        )

        for group, result in zip(df.ext.groupby_qid, results.ext.groupby_qid):
            grouped_uris.append(group.ext.uris)
            grouped_scores.append(result["scores"].values)

        return compute_letor_scores(grouped_uris, grouped_scores, targets)


class HandcraftedRankNetLetorModel(LetorModel):

    def __init__(self, number_of_features: int):
        super().__init__()
        self.number_of_features = number_of_features

    def fit_evaluate(self, df_train: pd.DataFrame, df_val: pd.DataFrame, verbose=False, n_epochs: int = 50) -> EvaluationResult:
        train_dataset = HandcraftedLetorDataset(df_train)
        train_loader = DataLoader(train_dataset, batch_size=64)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = "cpu"

        model = RankNet(self.number_of_features, device)

        trainer = RankNetTrainer(model, train_loader, df_val, device)
        trainer.train(n_epochs)

        self.model = trainer._model

        return self.evaluate(df_val)

    def evaluate(self, df_val: pd.DataFrame, verbose=False) -> EvaluationResult:
        return self.model.evaluate(df_val)

    @torch.no_grad()
    def rank(self, group: pd.DataFrame) -> List[str]:
        scores = self.model(torch.from_numpy(group.ext.X).float()).numpy()
        idx = np.argsort(scores)[::-1]
        return list(idx)


class RankNetWithEmbeddings(nn.Module):

    def __init__(self, device="cpu"):
        super(RankNetWithEmbeddings, self).__init__()

        self._device = device

        fasttext_embedding = WordEmbeddings('en-news')
        # flair_embedding_forward = FlairEmbeddings('news-forward')
        # flair_embedding_backward = FlairEmbeddings('news-backward')
        byte_pair_embedding = BytePairEmbeddings('en')
        glove_embeddings = WordEmbeddings('glove')
        character_embedding = CharacterEmbeddings()

        self._mention_embedding = DocumentPoolEmbeddings([fasttext_embedding])
        self._label_embedding = DocumentPoolEmbeddings([fasttext_embedding, ])
        self._context_embedding = DocumentPoolEmbeddings([fasttext_embedding])
        self._description_embedding = DocumentPoolEmbeddings([fasttext_embedding, ])

        input_length =   self._mention_embedding.embedding_length \
                       + self._context_embedding.embedding_length \
                       + self._label_embedding.embedding_length   \
                       + self._description_embedding.embedding_length

        self.model = nn.Sequential(
            nn.Linear(input_length, 256),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.output_sig = nn.Sigmoid()
        self.to(device)

    def score_pair(self, batch: Dict[str, Union[torch.Tensor, Sentence]]):
        mentions = batch["mention"]
        context = batch["context"]

        label_p = batch["label_p"]
        description_p = batch["description_p"]

        label_n = batch["label_n"]
        description_n = batch["description_n"]

        s1 = self.forward(mentions, label_p, description_p, context)
        s2 = self.forward(mentions, label_n, description_n, context)

        out = self.output_sig(s1 - s2)

        return out

    def forward(self, mentions: List[Sentence], labels: List[Sentence], descriptions: List[Sentence], contexts: List[Sentence]) -> torch.FloatTensor:
        mentions_tensor = self._embed(self._mention_embedding, mentions)
        labels_tensor = self._embed(self._label_embedding, labels)
        descriptions_tensor = self._embed(self._description_embedding, descriptions)
        contexts_tensor = self._embed(self._context_embedding, contexts)

        x = torch.cat([mentions_tensor, labels_tensor, descriptions_tensor, contexts_tensor], dim=1).to(self._device)

        return self.model(x)

    def _embed(self, embedding: Embeddings, sentences: List[Sentence]) -> torch.Tensor:
        embedding.embed(sentences)
        return torch.stack([sentence.embedding
                            # if len(sentence.tokens) else torch.zeros((embedding.embedding_length)).unsqueeze(0)
                            for sentence in sentences])

    @torch.no_grad()
    def evaluate(self, df: pd.DataFrame) -> EvaluationResult:
        grouped_scores = []
        grouped_uris = []
        targets = df.ext.gold_uris

        for group in tqdm(df.ext.groupby_qid, leave=False, desc="Evaluation"):
            mentions = [Sentence(x, use_tokenizer=False) for x in group["mention"]]
            labels = [Sentence(x, use_tokenizer=True) for x in group["label"]]
            descriptions = [Sentence(x, use_tokenizer=True) for x in group["description"]]
            contexts = [Sentence(x, use_tokenizer=False) for x in group["context"]]

            scores = self.forward(mentions, labels, descriptions, contexts).detach().squeeze(-1).numpy()
            grouped_scores.append(scores)
            grouped_uris.append(group.ext.uris)

        return compute_letor_scores(grouped_uris, grouped_scores, targets)


class RankNetWithEmbeddingsLetorModel(LetorModel):

    def fit_evaluate(self, df_train: pd.DataFrame, df_val: pd.DataFrame, verbose=False) -> EvaluationResult:
        train_dataset = PairwiseFlairLetorDataset(df_train)
        train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=my_collate)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = RankNetWithEmbeddings()
        model = model.to(device)
        self.model = model

        trainer = RankNetTrainer(model, train_loader, df_val, device)
        trainer.train(n_epochs=10)

        return self.evaluate(df_val)

    def evaluate(self, df_val: pd.DataFrame, verbose=False) -> EvaluationResult:
        return self.model.evaluate(df_val)


class RankNetTrainer:

    def __init__(self, model: nn.Module, train_loader: DataLoader, val_data, device):
        self._model = model
        self._train_loader = train_loader
        self._val_data = val_data

        self._device = device

    def train(self, n_epochs: int):
        optimizer = torch.optim.Adam(self._model.parameters())
        criterion = nn.BCELoss()
        early_stopping = EarlyStopping(self._model, "ranknet", mode="max", patience=5)

        for epoch in range(1, n_epochs + 1):
            epoch_loss = 0

            # Training
            for batch in tqdm(self._train_loader, leave=False, desc=f"Episode {epoch}"):
                y_p = batch["y_p"].to(self._device)
                y_n = batch["y_n"].to(self._device)

                score = self._model.score_pair(batch)

                loss = criterion(score, torch.sign(y_p - y_n))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()

            with torch.no_grad():
                validation_result: EvaluationResult = self._model.evaluate(self._val_data)

                if early_stopping.step(validation_result.accuracy_top5):
                    print(f"Early stopping", validation_result)
                    self._model = early_stopping.load_best_model()
                    break

            print(f"\rEpisode {epoch} - Validation Scores: ", validation_result)


def my_collate(batch):
    result = defaultdict(list)
    for e in batch:
        for k, v in e.items():
            result[k].append(v)

    return result
