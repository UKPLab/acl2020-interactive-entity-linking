import os

import numpy as np

import torch

from gleipnir.config import PATH_MODELS


# https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
# https://github.com/Bjarten/early-stopping-pytorch
# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    def __init__(self, model, name, mode='min', min_delta=0, patience=2, percentage=False):
        self.model = model
        self.name = name
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics) -> bool:
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            self._save_checkpoint()
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

    def _save_checkpoint(self):
        os.makedirs(PATH_MODELS, exist_ok=True)

        path = os.path.join(PATH_MODELS, f'checkpoint_{self.name}.pt')
        torch.save(self.model, path)

    def load_best_model(self):
        path = os.path.join(PATH_MODELS, f'checkpoint_{self.name}.pt')
        return torch.load(path)
