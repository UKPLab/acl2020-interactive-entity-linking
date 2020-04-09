'''

Uses stochastic variational inference (SVI) to scale to larger datasets with limited memory. At each iteration
of the VB algorithm, only a fixed number of random data points are used to update the distribution.

'''

import numpy as np
import logging

import scipy

from gleipnir.models.gppl.gp_classifier_vb import GPClassifierVB, max_no_jobs
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed
import multiprocessing
from scipy.special import psi


def _gradient_terms_for_subset(K_mm, invK_mm, kernel_derfactor, kernel_operator, common_term, ls_d, coords, s):

    if kernel_operator == '*':
        dKdls = K_mm * kernel_derfactor(coords, coords, ls_d, operator=kernel_operator) / s
    elif kernel_operator == '+':
        dKdls = kernel_derfactor(coords, coords, ls_d, operator=kernel_operator) / s

    return 0.5 * np.trace(common_term.dot(dKdls).dot(invK_mm * s) )


class GPClassifierSVI(GPClassifierVB):
    data_idx_i = []  # data indices to update in the current iteration, i
    changed_selection = True  # indicates whether the random subset of data has changed since variables were initialised
    covpair = None
    covpair_out = None

    def __init__(self, ninput_features, z0=0.5, shape_s0=2, rate_s0=2, shape_ls=10, rate_ls=0.1, ls_initial=None,
                 kernel_func='matern_3_2', kernel_combination='*', max_update_size=1000, ninducing=500, use_svi=True,
                 delay=1.0, forgetting_rate=0.9, verbose=False, fixed_s=False):

        self.max_update_size = max_update_size  # maximum number of data points to update in each SVI iteration

        # initialise the forgetting rate and delay for SVI
        self.forgetting_rate = forgetting_rate
        self.delay = delay  # delay must be at least 1

        # number of inducing points
        self.ninducing = ninducing

        self.n_converged = 10  # usually needs more converged iterations and can drop below zero due to approx. errors

        # default state before initialisation, unless some inducing coordinates are set by external call
        self.inducing_coords = None
        self.K_mm = None
        self.invK_mm = None
        self.K_nm = None
        self.K_star_m_star = None
        self.V_nn = None

        # if use_svi is switched off, we revert to the standard (parent class) VB implementation
        self.use_svi = use_svi

        self.reset_inducing_coords = True  # creates new inducing coords each time fit is called, if this flag is set

        self.exhaustive_train = 1
        # number of iterations that all training data must be used in when doing stochastic
        # sampling. You will need this setting on if you have any diagonal kernels/no person features.
        # Switching it off means that the algorithm will decide when to stop the stochastic updates.
        # It may think it has converged before seeing all the data.

        self.data_splits = None
        self.nsplits = 0 # we set this when data is passed in
        self.current_data_split = -1

        super(GPClassifierSVI, self).__init__(ninput_features, z0, shape_s0, rate_s0, shape_ls, rate_ls, ls_initial,
                                      kernel_func, kernel_combination, verbose=verbose, fixed_s=fixed_s)

    # Initialisation --------------------------------------------------------------------------------------------------

    def _init_params(self, mu0=None, reinit_params=True, K=None):
        if self.use_svi:
            self.update_size = self.max_update_size  # number of inducing points in each stochastic update
            if self.update_size > self.n_obs:
                self.update_size = self.n_obs
                # setting the forgetting rate to 0 here because we don't need to do stochastic updates
                self.current_forgetting_rate = 0
            else:
                self.current_forgetting_rate = self.forgetting_rate

            # in the first iteration with this dataset...
            if reinit_params or self.K_mm is None or self.vb_iter == 0:
                self._choose_inducing_points()

        super(GPClassifierSVI, self)._init_params(mu0, reinit_params, K)

    def _init_covariance(self):
        if not self.use_svi:
            return super(GPClassifierSVI, self)._init_covariance()

        self.obs_v = np.ones((self.n_locs, 1)) * self.rate_s0 / self.shape_s0

    def _init_s(self):
        if not self.use_svi:
            return super(GPClassifierSVI, self)._init_s()

        if not self.fixed_s:
            self.shape_s = self.shape_s0 + self.ninducing / 2.0
            self.rate_s = self.rate_s0 + 0.5 * (np.sum((self.obs_f-self.mu0)**2) + self.ninducing*self.rate_s0/self.shape_s0)

        self.s = self.shape_s / self.rate_s
        self.Elns = psi(self.shape_s) - np.log(self.rate_s)
        self.old_s = self.s
        if self.verbose:
            logging.debug("Setting the initial precision scale to s=%.3f" % self.s)

    def reset_kernel(self):
        self._init_covariance()
        if self.use_svi:
            self.K_mm = None
            self.K_nm = None
            self.K_star_m_star = None
            self.invK_mm = None

    def _choose_inducing_points(self):
        # choose a set of inducing points -- for testing we can set these to the same as the observation points.
        # diagonal can't use inducing points but can use the subsampling of observations
        if self.inducing_coords is None and (self.ninducing >= self.n_locs or self.cov_type == 'diagonal'):
            if self.inducing_coords is not None:
                logging.warning(
                    'replacing initial inducing points with observation coordinates because they are smaller.')
            self.ninducing = self.n_locs
            self.inducing_coords = self.obs_coords
            # invalidate matrices passed in to init_inducing_points() as we need to recompute for new inducing points
            self.reset_kernel()
        elif self.inducing_coords is None:
            init_size = 300
            if self.ninducing > init_size:
                init_size = self.ninducing
            kmeans = MiniBatchKMeans(init_size=init_size, n_clusters=self.ninducing)

            if self.obs_coords.shape[0] > 20 * self.ninducing:
                coords = self.obs_coords[np.random.choice(self.obs_coords.shape[0], 20 * self.ninducing, replace=False),
                         :]
            else:
                coords = self.obs_coords

            kmeans.fit(coords / self.ls[None, :])
            #
            self.inducing_coords = kmeans.cluster_centers_ * self.ls[None, :]

            # shuffled_idxs = np.random.permutation(self.obs_coords.shape[0])
            # self.inducing_coords = self.obs_coords[
            #     shuffled_idxs[
            #         np.unique(kmeans.labels_[shuffled_idxs], return_index=True)
            #         [1]]
            # ]
            # if self.inducing_coords.shape[0] < self.ninducing:
            #     self.ninducing = self.inducing_coords.shape[0]

            # self.inducing_coords = self.obs_coords[np.random.choice(self.n_locs, self.ninducing, replace=False), :]

            self.reset_kernel()

        if self.K_mm is None:
            self.K_mm = self.kernel_func(self.inducing_coords, self.ls, operator=self.kernel_combination, n_threads=self.n_threads)
            self.K_mm += 1e-6 * np.eye(len(self.K_mm))  # jitter

            self.Ks_mm = self.K_mm / self.s
        if self.invK_mm is None:
            if self.cov_type == 'diagonal':
                self.invK_mm = self.K_mm
            else:
                self.invK_mm = scipy.linalg.inv(self.K_mm)

            self.invKs_mm = self.invK_mm * self.s
        if self.K_nm is None:
            if self.cov_type == 'diagonal':
                self.K_nm = self.K_mm # there are no inducing points
            else:
                self.K_nm = self.kernel_func(self.obs_coords, self.ls, self.inducing_coords,
                                         operator=self.kernel_combination, n_threads=self.n_threads)
            self.Ks_nm = self.K_nm / self.s
        if not self.fixed_s:
            self.shape_s = self.shape_s0 + 0.5 * self.ninducing  # update this because we are not using n_locs data points

        # self.u_invSm = np.zeros((self.ninducing, 1), dtype=float)  # theta_1

        if self.cov_type == 'diagonal':
            self.u_invS = self.invK_mm * self.shape_s0 / self.rate_s0
            # self.u_invS = np.zeros((self.ninducing), dtype=float)  # theta_2
            self.u_Lambda = np.zeros((self.ninducing), dtype=float) # observation precision at inducing points
        else:
            self.u_invS = self.invK_mm * self.shape_s0 / self.rate_s0
            # self.u_invS = np.zeros((self.ninducing, self.ninducing), dtype=float)  # theta_2
            self.u_Lambda = np.zeros((self.ninducing, self.ninducing),
                                     dtype=float)  # observation precision at inducing points

        self.uS = self.K_mm * self.rate_s0 / self.shape_s0  # initialise properly to prior
        self.um_minus_mu0 = np.zeros((self.ninducing, 1))

        self.u_invSm = self.u_invS.dot(self.um_minus_mu0)

    # Mapping between latent and observation spaces -------------------------------------------------------------------

    def _compute_jacobian(self, f=None, data_idx_i=None):

        if f is None:
            f = self.obs_f

        if data_idx_i is not None:
            g_obs_f = self.forward_model(f.flatten()[data_idx_i])  # first order Taylor series approximation
        else:
            # if self.verbose:
            #     logging.debug("in _compute_jacobian, applying forward model to all observation points")
            g_obs_f = self.forward_model(f.flatten())
            # if self.verbose:
            #     logging.debug("in _compute_jacobian, computing gradients for all observation points...")
        J = np.diag(g_obs_f * (1 - g_obs_f))
        return g_obs_f, J

    def _update_jacobian(self, G_update_rate=1.0):
        if not self.use_svi:
            return super(GPClassifierSVI, self)._update_jacobian(G_update_rate)

        g_obs_f, J = self._compute_jacobian(data_idx_i=self.data_idx_i)

        if G_update_rate == 1 or not len(self.G) or self.G.shape != J.shape or self.changed_selection:
            # either G has not been initialised, or is from different observations, or random subset of data has changed
            self.G = J
        else:
            self.G = G_update_rate * J + (1 - G_update_rate) * self.G

        # set the selected observations i.e. not their locations, but the actual indexes in the input data. In the
        # standard case, these are actually the same anyway, but this can change if the observations are pairwise prefs.
        self.data_obs_idx_i = self.data_idx_i
        return g_obs_f

    # Log Likelihood Computation -------------------------------------------------------------------------------------

    def _logpt(self):
        logrho, lognotrho, _ = self._post_sample(self.obs_f, self.obs_v, expectedlog=True)

        return logrho, lognotrho

    def _logpf(self):
        # Note that some terms are cancelled with those in data_ll to simplify
        if not self.use_svi:
            return super(GPClassifierSVI, self)._logpf()

        _, logdet_K = np.linalg.slogdet(self.K_mm)
        D = len(self.um_minus_mu0)
        logdet_Ks = - D * self.Elns + logdet_K

        invK_expecF = np.trace(self.invKs_mm.dot(self.uS))

        m_invK_m = self.um_minus_mu0.T.dot(self.invK_mm * self.s).dot(self.um_minus_mu0)

        return 0.5 * (- np.log(2 * np.pi) * D - logdet_Ks - invK_expecF - m_invK_m)

    def _logqf(self):
        if not self.use_svi:
            return super(GPClassifierSVI, self)._logqf()

        # We want to do this, but we can simplify it, since the x and mean values cancel:
        _, logdet_C = np.linalg.slogdet(self.uS)
        D = len(self.um_minus_mu0)
        _logqf = 0.5 * (- np.log(2 * np.pi) * D - logdet_C - D)
        return _logqf

    def get_obs_precision(self):
        if not self.use_svi:
            return super(GPClassifierSVI, self).get_obs_precision()
        # _, G = self._compute_jacobian()
        # Lambda_factor1 = self.invKs_mm.dot(self.Ks_nm.T).dot(G.T)
        # Lambda_i = (Lambda_factor1 / self.Q[np.newaxis, :]).dot(Lambda_factor1.T)
        # return Lambda_i

        # this is different from the above because it is a weighted sum of previous values
        # return self.u_invS - (self.invKs_mm)

        if self.cov_type == 'diagonal':
            return np.diag(self.u_Lambda)
        return self.u_Lambda

    def lowerbound_gradient(self, dim):
        '''
        Gradient of the lower bound on the marginal likelihood with respect to the length-scale of dimension dim.
        '''
        if not self.use_svi:
            return super(GPClassifierSVI, self).lowerbound_gradient(dim)

        common_term = (self.um_minus_mu0.dot(self.um_minus_mu0.T) + self.uS).dot(self.s * self.invK_mm) - np.eye(self.ninducing)

        if self.n_lengthscales == 1 or dim == -1:  # create an array with values for each dimension
            dims = range(self.obs_coords.shape[1])
        else:  # do it for only the dimension dim
            dims = [dim]

        num_jobs = multiprocessing.cpu_count()
        if num_jobs > max_no_jobs:
            num_jobs = max_no_jobs
        if len(self.ls) > 1:
            gradient = Parallel(n_jobs=num_jobs, backend='threading')(
                delayed(_gradient_terms_for_subset)(self.K_mm, self.invK_mm, self.kernel_derfactor, self.kernel_combination,
                    common_term, self.ls[dim], self.inducing_coords[:, dim:dim + 1], self.s)
                for dim in dims)

        else:
            gradient = Parallel(n_jobs=num_jobs, backend='threading')(
                delayed(_gradient_terms_for_subset)(self.K_mm, self.invK_mm, self.kernel_derfactor, self.kernel_combination,
                    common_term, self.ls[0], self.inducing_coords[:, dim:dim + 1], self.s)
                for dim in dims)

        gradient *= self.ls

        if self.n_lengthscales == 1:
            # sum the partial derivatives over all the dimensions
            gradient = [np.sum(gradient)]

        return np.array(gradient)

    # Training methods ------------------------------------------------------------------------------------------------

    def _expec_f(self):
        if self.use_svi:
            # change the randomly selected observation points
            self._update_sample()

        super(GPClassifierSVI, self)._expec_f()

    def _update_f(self):
        if not self.use_svi:
            return super(GPClassifierSVI, self)._update_f()

        # this is done here not update_sample because it needs to be updated every time obs_f is updated
        self.obs_f_i = self.obs_f[self.data_idx_i]

        K_nm_i = self.K_nm[self.data_idx_i, :]

        Q = self.Q[self.data_obs_idx_i][np.newaxis, :]
        Lambda_factor1 = self.G.dot(K_nm_i).dot(self.invK_mm).T
        Lambda_i = (Lambda_factor1 / Q).dot(Lambda_factor1.T)

        if self.cov_type == 'diagonal':
            Lambda_i = np.diag(Lambda_i)

        # calculate the learning rate for SVI
        rho_i = (self.vb_iter + self.delay) ** (-self.current_forgetting_rate)
        # print("\rho_i = %f " % rho_i

        # weighting. Lambda and
        w_i = np.sum(self.obs_total_counts) / float(
            np.sum(self.obs_total_counts[self.data_obs_idx_i]))  # self.obs_f.shape[0] / float(self.obs_f_i.shape[0])

        # S is the variational covariance parameter for the inducing points, u. Canonical parameter theta_2 = -0.5 * S^-1.
        # The variational update to theta_2 is (1-rho)*S^-1 + rho*Lambda. Since Lambda includes a sum of Lambda_i over
        # all data points i, the stochastic update weights a sample sum of Lambda_i over a mini-batch.
        Lambda_i = Lambda_i * w_i * rho_i
        if self.cov_type == 'diagonal':
            self.u_invS = (1 - rho_i) * self.prev_u_invS + Lambda_i + rho_i * np.diag(self.invKs_mm)
        else:
            self.u_invS = (1 - rho_i) * self.prev_u_invS + Lambda_i + rho_i * self.invKs_mm
        self.u_Lambda = (1 - rho_i) * self.prev_u_Lambda + Lambda_i

        # use the estimate given by the Taylor series expansion
        z0 = self.forward_model(self.obs_f, subset_idxs=self.data_obs_idx_i) + self.G.dot(self.mu0_i - self.obs_f_i)
        y = self.z_i - z0

        # Variational update to theta_1 is (1-rho)*S^-1m + rho*beta*K_mm^-1.K_mn.y
        self.u_invSm = (1 - rho_i) * self.prev_u_invSm + w_i * rho_i * (Lambda_factor1 / Q).dot(y)

        # Next step is to use this to update f, so we can in turn update G. The contribution to Lambda_m and u_inv_S should therefore be made only once G has stabilised!
        # L_u_invS = cholesky(self.u_invS.T, lower=True, check_finite=False)
        # B = solve_triangular(L_u_invS, self.invKs_mm.T, lower=True, check_finite=False)
        # A = solve_triangular(L_u_invS, B, lower=True, trans=True, check_finite=False, overwrite_b=True)

        if self.cov_type == 'diagonal':
            self.uS = np.diag(1.0 / self.u_invS)
        else:
            self.uS = scipy.linalg.inv(self.u_invS)

        #         self.um_minus_mu0 = solve_triangular(L_u_invS, self.u_invSm, lower=True, check_finite=False)
        #         self.um_minus_mu0 = solve_triangular(L_u_invS, self.um_minus_mu0, lower=True, trans=True, check_finite=False,
        #                                              overwrite_b=True)
        self.um_minus_mu0 = self.uS.dot(self.u_invSm)

        if self.covpair is None:
            if self.cov_type == 'diagonal':
                self.covpair = 1.0
            else:
                self.covpair = scipy.linalg.solve(self.Ks_mm, self.Ks_nm.T).T

        self.obs_f, self.obs_v = self._f_given_u(self.covpair, self.mu0, 1.0 / self.s, full_cov=False)


    def _f_given_u(self, covpair, mu0, Ks_nn=None, full_cov=True):
        # see Hensman, Scalable variational Gaussian process classification, equation 18

        #(self.K_nm / self.s).dot(self.s * self.invK_mm).dot(self.uS).dot(self.u_invSm)
        if self.cov_type == 'diagonal':
            if self.um_minus_mu0.size != mu0.size:
                logging.error('We cannot make predictions for new test items when using a diagonal covariance -- we '
                              'need to be able to use the features to make predictions.')
                if Ks_nn is not None:
                    return np.zeros(mu0.size), np.diag(np.ones(mu0.size)/self.s)
                else:
                    return np.zeros(mu0.size)

            fhat = self.um_minus_mu0 + mu0
            if Ks_nn is not None and full_cov:
                C = self.uS
                return fhat, C
            elif Ks_nn is not None:
                C = np.diag(self.uS)[:, None]
                return fhat, C
            else:
                return fhat

        # for non-diagonal covariance matrices
        fhat = covpair.dot(self.um_minus_mu0) + mu0

        if Ks_nn is not None:
            if full_cov:
                C = Ks_nn + covpair.dot(self.uS - self.Ks_mm).dot(covpair.T)
                v = np.diag(C)
            else:
                C = Ks_nn + np.sum(covpair.dot(self.uS - self.Ks_mm) * covpair, axis=1)
                v = C
                C = C[:, None]

            if np.any(v < 0):
                logging.error("Negative variance in _f_given_u(), %f" % np.min(v))
                # caused by the accumulation of small errors due to stochastic updates. Can occur if s decreases in
                # later iterations - perhaps s needs stochastic updates to match?

                if full_cov:
                    fixidxs = np.argwhere(v < 0).flatten()
                    C[fixidxs, fixidxs] = 1e-6 # set to small number.
                else:
                    C[C<0] = 1e-6

            return fhat, C
        else:
            return fhat

    def _expec_s(self):
        if not self.use_svi:
            return super(GPClassifierSVI, self)._expec_s()

        self.old_s = self.s
        if self.cov_type == 'diagonal':
            invK_mm_expecFF = self.uS + self.um_minus_mu0.dot(self.um_minus_mu0.T)
        else:
            invK_mm_expecFF = self.invK_mm.dot(self.uS + self.um_minus_mu0.dot(self.um_minus_mu0.T))

        self.rate_s = self.rate_s0 + 0.5 * np.trace(invK_mm_expecFF)
        # Update expectation of s. See approximations for Binary Gaussian Process Classification, Hannes Nickisch
        self.s = self.shape_s / self.rate_s
        self.Elns = psi(self.shape_s) - np.log(self.rate_s)
        if self.verbose:
            logging.debug("Updated inverse output scale: " + str(self.s))

        self.Ks_mm = self.K_mm / self.s
        self.invKs_mm = self.invK_mm * self.s
        self.Ks_nm = self.K_nm / self.s

    def _update_sample(self):

        # once the iterations over G are complete, we accept this stochastic VB update
        self.prev_u_invSm = self.u_invSm
        self.prev_u_invS = self.u_invS
        self.prev_u_Lambda = self.u_Lambda

        self._update_sample_idxs()

        self.Ks_mm = self.K_mm / self.s
        self.invKs_mm = self.invK_mm * self.s
        self.Ks_nm = self.K_nm / self.s

        #self.G = 0  # reset because we will need to compute afresh with new sample. This shouldn't be necessary
        self.z_i = self.z[self.data_obs_idx_i]
        self.mu0_i = self.mu0[self.data_idx_i]

    def init_inducing_points(self, inducing_coords, K_mm=None, invK_mm=None, K_nm=None, V_nn=None):
        self.ninducing = inducing_coords.shape[0]
        self.inducing_coords = inducing_coords
        if K_mm is not None:
            self.K_mm = K_mm
        if invK_mm is not None:
            self.invK_mm = invK_mm
        if K_nm is not None:
            self.K_nm = K_nm
            self.K_star_m_star = None
        if V_nn is not None:
            self.V_nn = V_nn # the prior variance at the observation data points

        if self.cov_type == 'diagonal':
            self.u_invS = self.K_mm * self.rate_s / self.shape_s
            # np.zeros((self.ninducing), dtype=float)  # theta_2
            self.u_Lambda = np.zeros((self.ninducing), dtype=float)  # observation precision at inducing points
        else:
            # self.u_invS = np.zeros((self.ninducing, self.ninducing), dtype=float)  # theta_2
            self.u_invS = self.K_mm * self.rate_s / self.shape_s
            self.u_Lambda = np.zeros((self.ninducing, self.ninducing), dtype=float) # observation precision at inducing points
        self.uS = self.K_mm * self.rate_s0 / self.shape_s0  # initialise properly to prior
        self.um_minus_mu0 = np.zeros((self.ninducing, 1))

        # self.u_invSm = np.zeros((self.ninducing, 1), dtype=float)  # theta_1
        self.u_invSm = self.uS.dot(self.um_minus_mu0)#np.zeros((self.ninducing, 1), dtype=float)  # theta_1

    def _update_sample_idxs(self):
        if self.n_obs <= self.update_size:
            # we don't need stochastic updates if the update size is larger than number of observations
            self.data_idx_i = np.arange(self.obs_f.size)
            self.data_obs_idx_i = np.arange(self.n_obs)
            return

        # do this in the first iteration
        if self.nsplits == 0:
            self.nsplits = int(np.ceil(self.n_obs / float(self.update_size)))

            if self.exhaustive_train:
                self.min_iter_VB = self.nsplits
                if self.max_iter_VB < self.min_iter_VB * self.exhaustive_train:
                    self.max_iter_VB = self.min_iter_VB * self.exhaustive_train

        # do this each time we reach the end of updating for all splits in the current set
        if self.data_splits is None or np.mod(self.current_data_split+1, self.nsplits) == 0:
            # create nsplits random splits -- shuffle data and split
            rand_order = np.random.permutation(self.n_obs)
            self.data_splits = []

            for n in range(self.nsplits):
                ending = self.update_size * (n + 1)
                if ending > self.n_obs:
                    ending = self.n_obs
                self.data_splits.append(rand_order[self.update_size * n:ending])

            self.current_data_split = 0
        else:
            self.current_data_split += 1

        self.data_idx_i = self.data_splits[self.current_data_split]
        self.data_obs_idx_i = self.data_idx_i

    # Prediction methods ---------------------------------------------------------------------------------------------
    #
    def _get_training_cov(self):
        if not self.use_svi:
            return super(GPClassifierSVI, self)._get_training_cov()
        # return the covariance matrix for training points to inducing points (if used) and the variance of the training points.
        if self.K is not None:
            return self.K_nm, self.K
        else:
            if self.K_star_m_star is None:
                self.K_star_m_star = self.K_nm.dot(self.invK_mm).dot(self.K_nm.T)
            return self.K_nm, self.K_star_m_star

    def _get_training_feats(self):
        if not self.use_svi:
            return super(GPClassifierSVI, self)._get_training_feats()
        return self.inducing_coords

    def _expec_f_output(self, Ks_starstar, Ks_star, mu0, full_cov=False, reuse_output_kernel=False):
        """
        Compute the expected value of f and the variance or covariance of f
        :param Ks_starstar: prior variance at the output points (scalar or 1-D vector), or covariance if full_cov==True.
        :param Ks_star: covariance between output points and training points
        :param mu0: prior mean for output points
        :param full_cov: set to True to compute the full posterior covariance between output points
        :return f, C: posterior expectation of f, variance or covariance of the output locations.
        """
        if not self.use_svi:
            return super(GPClassifierSVI, self)._expec_f_output(Ks_starstar, Ks_star, mu0, full_cov, reuse_output_kernel)

        if self.covpair_out is None or not reuse_output_kernel:
            covpair_out = scipy.linalg.solve(self.K_mm/self.s, Ks_star.T).T
            if reuse_output_kernel:
                self.covpair_out = covpair_out
        elif reuse_output_kernel:
            covpair_out = self.covpair_out

        f, C_out = self._f_given_u(covpair_out, mu0, Ks_starstar, full_cov=full_cov)

        return f, C_out
