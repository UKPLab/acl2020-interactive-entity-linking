import logging

import numpy as np
import scipy
from scipy.stats import norm
from gp_classifier_svi import GPClassifierSVI


class GPRegressorSVI(GPClassifierSVI):

    # Input data handling ---------------------------------------------------------------------------------------------


    def _process_observations(self, obs_coords, obs_values, totals=None):
        if obs_values is None:
            return [], []

        self.obs_values = np.array(obs_values)

        obs_coords = np.array(obs_coords)
        if obs_coords.shape[0] == self.ninput_features and obs_coords.shape[1] != self.ninput_features:
            if obs_coords.ndim == 3 and obs_coords.shape[2] == 1:
                obs_coords = obs_coords.reshape((obs_coords.shape[0], obs_coords.shape[1]))
            obs_coords = obs_coords.T

        # assume one observation per location
        self.obs_coords = obs_coords
        self.obs_uidxs = np.arange(self.obs_coords.shape[0])
        self.n_obs = self.obs_coords.shape[0]

        if self.verbose:
            logging.debug("GP inference with %i observed data points." % self.n_obs)

        self.n_locs = self.obs_coords.shape[0]

        if self.verbose:
            logging.debug("Number of observed locations =" + str(self.obs_values.shape[0]))

        self.z = self.obs_values

        self.K_out = {}  # store the output cov matrix for each block. Reset when we have new observations.
        self.K_star_diag = {}

    def _init_obs_f(self):
        self.obs_f = np.copy(self.obs_values)

    def fit(self, obs_coords=None, obs_values=None, process_obs=True, mu0=None, K=None, optimize=False,
            maxfun=20, use_MAP=False, nrestarts=1,use_median_ls=False, obs_noise=None):

        if obs_noise is not None:
            self.Q = obs_noise

        super(GPRegressorSVI, self).fit(obs_coords, obs_values, None, process_obs, mu0, K, optimize, maxfun, use_MAP,
                                        nrestarts, None, use_median_ls)


    def estimate_obs_noise(self):
        """

        :param mu0: pass in the original mu0 here so we can use more efficient computation if it is scalar
        :return:
        """
        if self.Q is not None:
            return

        mu0 = self.mu0_input
        if np.isscalar(mu0):
            n_locs = 1  # sample only once, and use the estimated values across all points
        else:
            n_locs = len(mu0)

        self.Q = np.zeros(n_locs) + 1e-10

    # Log Likelihood Computation -------------------------------------------------------------------------------------

    def _logp_Df(self):
        logdll = np.sum(norm.logpdf(self.obs_values, self.obs_f.flatten(), self.Q))\
                 - 0.5 * np.sum(self.obs_v.flatten() / self.Q)

        logpf = self._logpf()
        return logpf + logdll

    # Training methods ------------------------------------------------------------------------------------------------

    def _expec_f(self):
        if self.use_svi:
            # change the randomly selected observation points
            self._update_sample()

        '''
        Compute the expected value of f given current q() distributions for other parameters. Could plug in a different
        GP implementation here.
        '''
        self._update_f()


    def _update_f(self):
        # this is done here not update_sample because it needs to be updated every time obs_f is updated
        self.obs_f_i = self.obs_f[self.data_idx_i]

        K_nm_i = self.K_nm[self.data_idx_i, :]

        Q = self.Q[self.data_obs_idx_i][np.newaxis, :]
        Lambda_factor1 = K_nm_i.dot(self.invK_mm).T
        Lambda_i = (Lambda_factor1 / Q).dot(Lambda_factor1.T)

        if self.cov_type == 'diagonal':
            Lambda_i = np.diag(Lambda_i)

        # calculate the learning rate for SVI
        rho_i = (self.vb_iter + self.delay) ** (-self.current_forgetting_rate)
        # print("\rho_i = %f " % rho_i

        # weighting. Lambda and
        w_i = self.n_locs / float(len(self.data_obs_idx_i))# self.obs_f.shape[0] / float(self.obs_f_i.shape[0])

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
        y = self.z_i[:, None] - self.mu0[self.data_idx_i]

        # Variational update to theta_1 is (1-rho)*S^-1m + rho*beta*K_mm^-1.K_mn.y
        self.u_invSm = (1 - rho_i) * self.prev_u_invSm + w_i * rho_i * (Lambda_factor1 / Q).dot(y)

        if self.cov_type == 'diagonal':
            self.uS = np.diag(1.0 / self.u_invS)
        else:
            self.uS = scipy.linalg.inv(self.u_invS)

        self.um_minus_mu0 = self.uS.dot(self.u_invSm)

        if self.covpair is None:
            if self.cov_type == 'diagonal':
                self.covpair = 1.0
            else:
                self.covpair = scipy.linalg.solve(self.Ks_mm, self.Ks_nm.T).T

        self.obs_f, self.obs_v = self._f_given_u(self.covpair, self.mu0, 1.0 / self.s, full_cov=False)