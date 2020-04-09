from typing import List

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from gleipnir.datasets import load_handcrafted_data
from gleipnir.evaluation.metrics import EvaluationResult, compute_letor_scores


class NegativeRanknet(nn.Module):
    """ Uses MultipleNegativesRankingLoss """

    def __init__(self, num_features: int):
        super().__init__()

        self._model = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, X: torch.FloatTensor) -> torch.Tensor:
        return self._model(X)

    def multiple_negatives_ranking_loss(self, positive_batch: List[torch.FloatTensor], negative_batch: List[torch.Tensor]):
        scores = []

        for X_p, X_n in zip(positive_batch, negative_batch):
            # Efficient Natural Language Response Suggestion for Smart Reply 4.4
            # https://arxiv.org/pdf/1705.00652.pdf
            positive_score = self(X_p)       # S(x_i, y_i)
            negative_scores = self(X_n)      # S(x_i, y_j)

            log_sum_exp = torch.logsumexp(negative_scores, dim=0)
            scores.append(positive_score - log_sum_exp)

        scores_tensor = torch.tensor(scores, dtype=torch.float, requires_grad=True)

        return - torch.mean(scores_tensor)

    @torch.no_grad()
    def evaluate(self, df: pd.DataFrame) -> EvaluationResult:
        grouped_scores = []
        grouped_uris = []
        targets = df.ext.gold_uris

        scores = self.forward(torch.from_numpy(df.ext.X).float()).squeeze().numpy()

        results = pd.DataFrame(
            {
                "scores": scores,
                "qid": df["qid"].values
            }
        )

        for group, result in zip(df.ext.groupby_qid, results.ext.groupby_qid):
            grouped_uris.append(group.ext.uris)
            grouped_scores.append(result["scores"].values)

        return compute_letor_scores(grouped_uris, grouped_scores, targets)


class GroupingHandcraftedDataset(Dataset):

    def __init__(self, df: pd.DataFrame):
        X_positive_grouped = []
        X_negative_grouped = []

        for group in df.ext.groupby_qid:
            gold_idx = group.ext.gold_indices[0]
            X = group.ext.X
            X_positive = X[gold_idx]
            X_negative = np.delete(X, gold_idx, axis=0)

            assert X_positive.shape == (df.ext.num_features, )
            assert X_negative.shape == (len(X) - 1, df.ext.num_features)

            X_positive_grouped.append(torch.from_numpy(X_positive))
            X_negative_grouped.append(torch.from_numpy(X_negative))

        assert len(X_positive_grouped) == len(X_negative_grouped)

        self._number_of_items = len(X_positive_grouped)
        self._X_positive_grouped = X_positive_grouped
        self._X_negative_grouped = X_negative_grouped

    def __len__(self) -> int:
        return self._number_of_items

    def __getitem__(self, group_idx: int):
        X_p = self._X_positive_grouped[group_idx]
        X_n = self._X_negative_grouped[group_idx]

        return X_p, X_n


def my_collate(batch):
    return list(zip(*batch))


if __name__ == '__main__':
    df_train, df_dev = load_handcrafted_data("aida")
    debug = True

    if debug:
        df_train = df_train.ext.subsample(100)
        df_dev = df_dev.ext.subsample(10)

    train_dataset = GroupingHandcraftedDataset(df_train)

    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=my_collate)

    model = NegativeRanknet(df_train.ext.num_features)
    optimizer = torch.optim.Adam(model.parameters())

    n_epochs = 10

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0

        # Training
        for batch in tqdm(train_loader, leave=False, desc=f"Episode {epoch}"):
            X_p, X_n = batch
            model.train()

            loss = model.multiple_negatives_ranking_loss(X_p, X_n)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        with torch.no_grad():
            validation_result: EvaluationResult = model.evaluate(df_dev)

        print(f"\rEpisode {epoch} - Loss: {epoch_loss} - Validation Scores: {validation_result}")


