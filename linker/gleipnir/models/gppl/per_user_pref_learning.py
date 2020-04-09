import multiprocessing

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm

from gp_pref_learning import GPPrefLearning


def usermodel_predict_f(model, item_features):
    if model.vb_iter == 0:
        # not trained, skip it
        fpredu = np.zeros((item_features.shape[0], 1))
    else:
        fpredu, _ = model.predict_f(item_features)

    return fpredu

class GPPrefPerUser():
    '''
    Runs a separate preference learning model for each user. I.e. multiple users but no collaborative learning.
    '''

    def __init__(self, Npeople, max_update_size, shape_s0, rate_s0, nitem_feats=2, ninducing=50, verbose=True, delay=1.0):
        self.user_models = []

        self.Npeople = Npeople
        for p in range(Npeople):
            model_p = GPPrefLearning(nitem_feats, mu0=0, shape_s0=shape_s0, rate_s0=rate_s0, ls_initial=None,
                                     use_svi=True, ninducing=ninducing,
                                     max_update_size=max_update_size, forgetting_rate=0.9, delay=delay, verbose=verbose)

            model_p.max_iter_VB /= Npeople # allow the same number of iterations in total as other methods.

            self.user_models.append(model_p)


    def fit(self, users, p1, p2, item_features, prefs, _, optimize, use_median_ls):

        uusers = np.unique(users)
        for u in uusers:
            uidxs = users == u

            self.user_models[u].fit(
                p1[uidxs],
                p2[uidxs],
                item_features,
                prefs[uidxs],
                optimize=optimize,
                use_median_ls=use_median_ls
            )

    def predict_f(self, item_features, person_features=None, personids=None):

        if personids is None:
            chosen_users = range(self.Npeople)
        else:
            chosen_users = personids

        num_jobs = multiprocessing.cpu_count()
        fpredus = Parallel(n_jobs=num_jobs, backend='threading')(delayed(usermodel_predict_f)(
            self.user_models[u], item_features) for u in chosen_users)

        fpred = np.concatenate(fpredus, axis=1)

        return fpred

    def predict(self, users, p1, p2, item_features, _):

        rhopred = np.zeros(len(p1))

        uusers = np.unique(users)
        for u in uusers:
            uidxs = users.flatten() == u

            if self.user_models[u].vb_iter == 0:
                # not trained, skip it
                rho_pred_u = np.zeros(p1[uidxs].shape[0])
            else:
                rho_pred_u, _ = self.user_models[u].predict(item_features, p1[uidxs], p2[uidxs])
            rhopred[uidxs] = rho_pred_u.flatten()

        return rhopred

    def predict_t(self, item_features):
        F = self.predict_f(item_features)
        return np.mean(F, axis=1)

    def predict_common(self, item_features, p1, p2):
        F = self.predict_f(item_features)

        # predict the common mean/consensus or underlying ground truth function
        g_f = (np.mean(F[p1], axis=1) - np.mean(F[p2], axis=1)).astype(int) / np.sqrt(2)
        phi = norm.cdf(g_f)
        return phi

    def lowerbound(self):
        lb = 0
        for model in self.user_models:
            lb += model.lowerbound()

        return lb