"""
Scalable implementation of collaborative Gaussian process preference learning using stochastic variational inference.
Scales to large sets of observations (preference pairs) and numbers of items and users.

The results look different to the non-SVI version. There is a difference in how G is computed inside expec_t --
in the non-SVI version, it is computed separately for each observation location, and the obs_f estimates used to compute
it do not have any shared component across people because the value of t is computed by aggregating across people
outside the child GP. With the SVI implementation, the aggregation is done at each step by the inducing points, so that
inside the iterations of t_gp, there is a common t value when computing obs_f for all people. I think both are valid
approximations considering they are using slightly different values of obs_f to compute the updates. Differences may
accumulate from small differences in the approximations.

"""
import datetime
import os

import numpy as np
from numpy.core.umath import square
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal as mvn, norm
import logging
from gp_pref_learning import pref_likelihood
from gp_classifier_vb import temper_extreme_probs
from scipy.special import psi, binom
from sklearn.cluster import MiniBatchKMeans

from collab_pref_learning_vb import CollabPrefLearningVB, expec_output_scale, expec_pdf_gaussian, expec_q_gaussian, \
    temper_extreme_probs, lnp_output_scale, lnq_output_scale


def svi_update_gaussian(invQi_y, mu0_n, mu_u, K_mm, invK_mm, K_nm, Lambda_factor1, K_nn, invQi, prev_invS, prev_invSm,
                        vb_iter, delay, forgetting_rate, N, update_size):
    Lambda_i = Lambda_factor1.dot(invQi).dot(Lambda_factor1.T)

    # calculate the learning rate for SVI
    rho_i = (vb_iter + delay) ** (-forgetting_rate)
    # print "\rho_i = %f " % rho_i

    # weighting. Lambda and
    w_i = N / float(update_size)

    # S is the variational covariance parameter for the inducing points, u.
    # Canonical parameter theta_2 = -0.5 * S^-1.
    # The variational update to theta_2 is (1-rho)*S^-1 + rho*Lambda. Since Lambda includes a sum of Lambda_i over
    # all data points i, the stochastic update weights a sample sum of Lambda_i over a mini-batch.
    invS = (1 - rho_i) * prev_invS + rho_i * (w_i * Lambda_i + invK_mm)

    # Variational update to theta_1 is (1-rho)*S^-1m + rho*beta*K_mm^-1.K_mn.y
    #     invSm = (1 - rho_i) * prev_invSm + w_i * rho_i * invK_mm.dot(K_im.T).dot(invQi).dot(y)
    invSm = (1 - rho_i) * prev_invSm + w_i * rho_i * Lambda_factor1.dot(invQi_y)

    # Next step is to use this to update f, so we can in turn update G. The contribution to Lambda_m and u_inv_S should therefore be made only once G has stabilised!
    # L_invS = cholesky(invS.T, lower=True, check_finite=False)
    # B = solve_triangular(L_invS, invK_mm.T, lower=True, check_finite=False)
    # A = solve_triangular(L_invS, B, lower=True, trans=True, check_finite=False, overwrite_b=True)
    # invK_mm_S = A.T
    S = np.linalg.inv(invS)
    invK_mm_S = invK_mm.dot(S)

    # fhat_u = solve_triangular(L_invS, invSm, lower=True, check_finite=False)
    # fhat_u = solve_triangular(L_invS, fhat_u, lower=True, trans=True, check_finite=False, overwrite_b=True)
    fhat_u = S.dot(invSm)
    fhat_u += mu_u

    # TODO: move the K_mm.T.dot(K_nm.T) computation out
    covpair_uS = K_nm.dot(invK_mm_S)
    fhat = covpair_uS.dot(invSm) + mu0_n
    if K_nn is None:
        C = None
    else:
        covpair = K_nm.dot(invK_mm)
        C = K_nn + (covpair_uS - covpair.dot(K_mm)).dot(covpair.T)
    return fhat, C, invS, invSm, fhat_u, invK_mm_S, S

def inducing_to_observation_moments(Ks_mm, invK_mm, K_nm, fhat_mm, mu0, S=None, Ks_nn=None):
    covpair = K_nm.dot(invK_mm)
    fhat = covpair.dot(fhat_mm) + mu0

    if S is None or Ks_nn is None:
        C = None
    else:
        C2 = Ks_nn + np.diag(covpair.dot(S-Ks_mm).dot(covpair.T))
        C = Ks_nn + np.sum(covpair.dot(S - Ks_mm) * covpair, axis=1)

    return fhat, C

class CollabPrefLearningSVI(CollabPrefLearningVB):

    def __init__(self, nitem_features, nperson_features=0, mu0=0, shape_s0=1, rate_s0=1, shape_st0=None, rate_st0=None,
                 shape_sy0=None, rate_sy0=None, shape_ls=1, rate_ls=100, ls=100, shape_lsy=1, rate_lsy=100, lsy=100,
                 verbose=False, nfactors=20, use_common_mean_t=True, kernel_func='matern_3_2',
                 max_update_size=500, ninducing=500, forgetting_rate=0.9, delay=1.0, use_lb=True,
                 exhaustive_train_count=1, personal_component=False):

        # if this is true, each user gets their own component, which uses item features. All other components and the
        # consensus will not use item features
        self.personal_component = personal_component

        self.max_update_size = max_update_size
        self.ninducing_preset = ninducing
        self.forgetting_rate = forgetting_rate
        self.delay = delay

        self.factors_with_features = None

        self.conv_threshold_G = 1e-5

        self.t_mu0 = mu0

        self.max_Kw_size = 500 # maximum size to hold in memory. Larger matrices are saved to memmap files

        super(CollabPrefLearningSVI, self).__init__(nitem_features, nperson_features, shape_s0, rate_s0, shape_st0,
            rate_st0, shape_sy0, rate_sy0, shape_ls, rate_ls, ls, shape_lsy, rate_lsy, lsy, verbose, nfactors,
            use_common_mean_t, kernel_func, use_lb=use_lb)

        self.exhaustive_train = exhaustive_train_count
        # number of iterations that all training data must be used in when doing stochastic
        # sampling. You will need this setting on if you have any diagonal kernels/no person features.
        # Switching it off means that the algorithm will decide when to stop the stochastic updates.
        # It may think it has converged before seeing all the data.

        self.data_splits = None
        self.nsplits = 0 # we set this when data is passed in
        self.current_data_split = -1


    def _init_covariance(self):
        self.shape_sw = np.zeros(self.Nfactors + (self.Npeople if self.personal_component else 0) ) + self.shape_sw0
        self.rate_sw = np.zeros(self.Nfactors + (self.Npeople if self.personal_component else 0) ) + self.rate_sw0

        self.shape_sy = np.zeros(self.Nfactors + (self.Npeople if self.personal_component else 0) ) + self.shape_sy0
        self.rate_sy = np.zeros(self.Nfactors + (self.Npeople if self.personal_component else 0) ) + self.rate_sy0

    def _choose_inducing_points(self):
        # choose a set of inducing points -- for testing we can set these to the same as the observation points.
        self.update_size = self.max_update_size # number of observed points in each stochastic update
        if self.update_size > self.nobs:
            self.update_size = self.nobs

        # Inducing points for items -----------------------------------------------------------

        self.ninducing = self.ninducing_preset

        if self.ninducing >= self.obs_coords.shape[0]:
            self.ninducing = self.obs_coords.shape[0]
            self.inducing_coords = self.obs_coords
        else:
            init_size = 300
            if init_size < self.ninducing:
                init_size = self.ninducing
            kmeans = MiniBatchKMeans(init_size=init_size, n_clusters=self.ninducing)
            kmeans.fit(self.obs_coords / self.ls[None, :])

            self.inducing_coords = kmeans.cluster_centers_ * self.ls[None, :]

            # shuffled_idxs = np.random.permutation(self.person_features.shape[0])
            #
            # self.inducing_coords = self.obs_coords[
            #     shuffled_idxs[
            #         np.unique(kmeans.labels_[shuffled_idxs], return_index=True)
            #         [1]]
            # ]
            #
            # if self.inducing_coords.shape[0] < self.ninducing:
            #     self.ninducing = self.inducing_coords.shape[0]

            # self.inducing_coords = self.obs_coords[np.random.choice(self.N, self.ninducing, replace=False), :]

        # Kernel over items (used to construct priors over w and t)
        if self.verbose:
            logging.debug('Initialising K_mm')
        self.K_mm = self.kernel_func(self.inducing_coords, self.ls) + \
                    (1e-4 if self.cov_type=='diagonal' else 1e-6) * np.eye(self.ninducing) # jitter
        self.invK_mm = np.linalg.inv(self.K_mm)
        if self.verbose:
            logging.debug('Initialising K_nm')
        self.K_nm = self.kernel_func(self.obs_coords, self.ls, self.inducing_coords)

        # Related to w, the item components ------------------------------------------------------------
        # posterior expected values
        # self.w_u = mvn.rvs(np.zeros(self.ninducing), self.K_mm, self.Nfactors)
        # self.w_u /= (self.shape_sw / self.rate_sw)[:, None]
        # self.w_u = self.w_u.T
        self.w_u = np.zeros((self.ninducing, self.Nfactors))
        #self.w_u = self.w_u[np.arange(self.ninducing), np.arange(self.ninducing)] = 1.0

        sy = self.shape_sy / self.rate_sy

        # Inducing points for people -------------------------------------------------------------------
        if self.person_features is None:

            self.y_ninducing = self.Npeople

            # Prior covariance of y
            self.Ky_mm = np.ones(self.y_ninducing)
            self.invKy_mm = self.Ky_mm
            self.Ky_nm = np.diag(self.Ky_mm)

            self.use_local_obs_posterior_y = False

            # posterior covariance
            #self.yS = np.zeros((self.Nfactors, self.y_ninducing))
            self.yS = np.array([self.Ky_mm/sy[f] for f in range(self.Nfactors + (self.Npeople if self.personal_component else 0) )])
            # self.yinvS = np.zeros((self.Nfactors, self.y_ninducing))
            self.yinvS = np.array([self.invKy_mm*sy[f] for f in range(self.Nfactors + (self.Npeople if self.personal_component else 0) )])

            self.y_u = norm.rvs(0, 1, (self.Nfactors, self.y_ninducing))**2

            self.yinvSm = np.zeros((self.Nfactors + (self.Npeople if self.personal_component else 0) , self.y_ninducing))
            #self.yinvSm = np.concatenate([(self.yinvS[f] * (self.y_u[f]))[:, None] for f in range(self.Nfactors)], axis=1)

        else:
            self.y_ninducing = self.ninducing_preset

            # When we use inducing points, we can assume that posterior predictions depend on both the inducing points and
            # the local observations at that point. This can be applied to predicting the latent person features so that
            # when the observed person features are only weakly informative, we can still use the local observations to
            # make predictions and don't rely on the inducing points. The same could be done for the items but has
            # not yet been implemented.
            #self.use_local_obs_posterior_y = True

            if self.y_ninducing >= self.Npeople:
                self.y_ninducing = self.Npeople
                self.y_inducing_coords = self.person_features
            else:
                init_size = 300
                if self.y_ninducing > init_size:
                    init_size = self.y_ninducing
                kmeans = MiniBatchKMeans(init_size=init_size, n_clusters=self.y_ninducing, compute_labels=True)
                kmeans.fit(self.person_features / self.lsy[None, :])

                shuffled_idxs = np.random.permutation(self.person_features.shape[0])

                self.y_inducing_coords = self.person_features[
                    shuffled_idxs[
                        np.unique(kmeans.labels_[shuffled_idxs], return_index=True)
                    [1]]
                ]

                if self.y_inducing_coords.shape[0] < self.y_ninducing:
                     self.y_ninducing = self.y_inducing_coords.shape[0]

                # self.y_inducing_coords = self.person_features[np.random.choice(self.Npeople,
                #                                                                self.y_ninducing,
                #                                                                replace=False)]
                # self.person_features[:self.y_ninducing]

            # Kernel over people used to construct prior covariance for y
            if self.verbose:
                logging.debug('Initialising Ky_mm')
            self.Ky_mm = self.y_kernel_func(self.y_inducing_coords, self.lsy)
            self.Ky_mm += (1e-4 if self.cov_type == 'diagonal' else 1e-6) * np.eye(self.y_ninducing) # jitter

            # Prior covariance of y
            self.invKy_mm = np.linalg.inv(self.Ky_mm)

            if self.verbose:
                logging.debug('Initialising Ky_nm')
            self.Ky_nm = self.y_kernel_func(self.person_features, self.lsy, self.y_inducing_coords)

            # posterior covariance
            #self.yS = np.zeros((self.Nfactors, self.y_ninducing, self.y_ninducing))

            if self.factors_with_features is None:
                self.factors_with_features = np.arange(self.Nfactors)

            self.yS = [self.Ky_mm/sy[f] if f in self.factors_with_features
                                else np.ones(self.Npeople)/sy[f] for f in range(self.Nfactors + (self.Npeople if self.personal_component else 0) )]
            # self.yinvS = np.zeros((self.Nfactors, self.y_ninducing, self.y_ninducing))
            self.yinvS = [self.invKy_mm*sy[f]  if f in self.factors_with_features
                                else np.ones(self.Npeople)/sy[f] for f in range(self.Nfactors + (self.Npeople if self.personal_component else 0) )]

            self.y_u = [norm.rvs(0, 1, self.y_ninducing) if f in self.factors_with_features
                                else norm.rvs(0, 1, self.Npeople) for f in range(self.Nfactors + (self.Npeople if self.personal_component else 0) )]


            self.yinvSm = [np.ones(self.y_ninducing) if f in self.factors_with_features else
                np.ones(self.Npeople) for f in range(self.Nfactors + (self.Npeople if self.personal_component else 0) )]
            #self.yinvSm = np.concatenate([self.yinvS[f].dot(self.y_u[f:f+1].T) for f in range(self.Nfactors)], axis=1)

        self.prev_yinvSm = self.yinvSm.copy()
        self.prev_yinvS = self.yinvS.copy()

        if self.Nfactors == 1 and self.y_u.ndim == 1:
            self.y_u = self.y_u[None, :]

        # Related to t, the item means -----------------------------------------------------------------
        self.t_u = np.zeros((self.ninducing, 1))  # posterior means
        self.tS = None

        if self.use_t:
            # self.tinvS = np.zeros((self.ninducing, self.ninducing))
            self.tinvS = self.invK_mm * self.shape_st0 / self.rate_st0 # theta_2/posterior covariance
            self.tinvSm = np.zeros((self.ninducing, 1))
            #self.tinvSm = self.tinvS.dot(self.t_u)


        # add personal components
        if self.personal_component:
            self.Nfactors += self.Npeople

            self.V = np.zeros((self.N, self.Npeople))

            dists = pdist(self.obs_coords / self.ls, metric='euclidean')
            self.Kv = np.exp( -0.5 * dists**2 )
            self.Kv = squareform(self.Kv)
            np.fill_diagonal(self.Kv, 1)

            self.invKv = np.linalg.inv(self.Kv)

            self.w_u = np.concatenate((self.w_u, self.V), axis=1)

        # moments of distributions over inducing points for convenience
        # posterior covariance
        self.wS = np.zeros((self.Nfactors, self.ninducing, self.ninducing))
        # self.wS = [self.K_mm * self.shape_sw0 / self.rate_sw0 for _ in range(self.Nfactors)])

        # self.winvS = np.zeros((self.Nfactors, self.ninducing, self.ninducing))
        self.winvS = np.array([self.invK_mm * self.shape_sw0 / self.rate_sw0 for _ in range(self.Nfactors)])

        self.winvSm = np.zeros((self.ninducing, self.Nfactors))
        # self.winvSm = np.concatenate([self.winvS[f].dot(self.w_u[:, f:f+1]) for f in range(self.Nfactors)], axis=1)
        self.prev_winvS = self.winvS.copy()
        self.prev_winvSm = self.winvSm.copy()


    def _post_sample(self, K_nm, invK_mm, w_u, wS, t_u, tS,
                     Ky_nm, invKy_mm, y_u, y_var, v, u, expectedlog=False):

        # sample the inducing points because we don't have full covariance matrix. In this case, f_cov should be Ks_nm
        nsamples = 100#0

        if wS.ndim == 3:
            w_samples = np.array([mvn.rvs(mean=w_u[:, f], cov=wS[f], size=(nsamples))
                              for f in range(self.Nfactors)])
        else:
            w_samples = np.array([mvn.rvs(mean=w_u[:, f], cov=wS, size=(nsamples))
                                  for f in range(self.Nfactors)])

        if self.use_t:
            if np.isscalar(t_u):
                t_u = np.zeros(tS.shape[0]) + t_u
            else:
                t_u = t_u.flatten()

            t_samples = mvn.rvs(mean=t_u, cov=tS, size=(nsamples))

        N = y_u.shape[1]
        if np.isscalar(y_var):
            y_var = np.zeros((self.Nfactors * N)) + y_var
        else:
            y_var = y_var.flatten()

        y_samples = np.random.normal(loc=y_u.flatten()[:, None], scale=np.sqrt(y_var)[:, None],
                                     size=(N * self.Nfactors, nsamples)).reshape(self.Nfactors, N, nsamples)

        # w_samples: F x nsamples x N
        # t_samples: nsamples x N
        # y_samples: F x Npeople x nsamples

        if K_nm is not None:
            covpair_w = K_nm.dot(invK_mm)
            w_samples = np.array([covpair_w.dot(w_samples[f].T).T
                for f in range(self.Nfactors)])
            # assume zero mean
            if self.use_t:
                t_samples = K_nm.dot(invK_mm).dot(t_samples.T).T

            if self.person_features is not None:
                covpair_y = Ky_nm.dot(invKy_mm)
                y_samples = np.array([covpair_y.dot(y_samples[f])
                      for f in range(self.Nfactors)])  # assume zero mean

        if self.use_t:
            f_samples = np.array([w_samples[:, s, :].T.dot(y_samples[:, :, s]).T + t_samples[s][None, :]for s in range(nsamples)])
        else:
            f_samples = np.array([w_samples[:, s, :].T.dot(y_samples[:, :, s]).T for s in range(nsamples)])

        f_samples = f_samples.reshape(nsamples, self.N * self.Npeople).T

        phi = pref_likelihood(f_samples, v=v, u=u)
        phi = temper_extreme_probs(phi)
        notphi = 1 - phi

        if expectedlog:
            phi = np.log(phi)
            notphi = np.log(notphi)

        m_post = np.mean(phi, axis=1)[:, np.newaxis]
        not_m_post = np.mean(notphi, axis=1)[:, np.newaxis]
        v_post = np.var(phi, axis=1)[:, np.newaxis]
        v_post = temper_extreme_probs(v_post, zero_only=True)
        # fix extreme values to sensible values. Don't think this is needed and can lower variance?
        v_post[m_post * (1 - not_m_post) <= 1e-7] = 1e-8

        return m_post, not_m_post, v_post

    def _estimate_obs_noise(self):

        # in the generative model, the value of f determines Q.

        # To make a and b smaller and put more weight onto the observations, increase v_prior by increasing rate_s0/shape_s0
        m_prior = 0.5
        _, _, v_prior = self._post_sample(self.K_nm, self.invK_mm,
                                  np.zeros((self.ninducing, self.Nfactors)), self.K_mm * self.rate_sw0 / self.shape_sw0,
                                  self.t_mu0, self.K_mm * self.rate_st0 / self.shape_st0,
                                  self.Ky_nm, self.invKy_mm,
                                  np.zeros((self.Nfactors, self.y_ninducing)), 1,#self.rate_sy0 / self.shape_sy0,
                                  self.pref_v, self.pref_u)

        # find the beta parameters
        a_plus_b = 1.0 / (v_prior / (m_prior*(1-m_prior))) - 1
        a = a_plus_b * m_prior
        b = a_plus_b * (1 - m_prior)

        nu0 = np.array([b, a])
        # Noise in observations
        nu0_total = np.sum(nu0, axis=0)
        obs_mean = (self.z + nu0[1]) / (1 + nu0_total)
        #var_obs_mean = obs_mean * (1 - obs_mean) / (1 + nu0_total + 1)  # uncertainty in obs_mean
        Q = obs_mean * (1 - obs_mean) #+ var_obs_mean
        Q = Q.flatten()

        # # this setup produced the good consensus conv scores. The good Sushi-B and personalised conv scores
        # # require switching the scale to 2.
        # f_samples = np.random.normal(loc=0, scale=np.sqrt(self.rate_st0 / self.shape_st0
        #                 + np.sum(self.rate_sw * self.rate_sy / (self.shape_sw * self.shape_sy)) ),
        #                 size=(self.N, 1000))
        #
        # phi = pref_likelihood(f_samples, v=self.tpref_v, u=self.tpref_u)
        # v_prior = np.var(phi)
        #
        # # the below is a very rough approximation because it ignores the covariance between points,
        # # hence m_prior values will be more extreme and v_prior likely larger.
        # # In practice this seemed to work well in the single user model and is quick to compute.
        # # Assume that each data point has the same prior...
        # a_plus_b = 1.0 / (v_prior / (m_prior*(1-m_prior))) - 1
        # a = a_plus_b * m_prior
        # b = a_plus_b * (1 - m_prior)
        #
        # nu0 = np.array([a, b])
        # # Noise in observations
        # nu0_total = np.sum(nu0, axis=0)
        # obs_mean = (self.z + nu0[1]) / (1 + nu0_total)
        # Q2 = obs_mean * (1 - obs_mean) # don't think variance is needed because the equations show it is E[p]^2 not E[p^2] since we have already taken expectations before we get to this point
        # Q = Q2.flatten()

        # TODO: note that the sqrt(2) version below was used to get the best SushiB and Conv-Personal results,
        # but does not achieve this any more for Sushi B when used alone. Therefore, we need to check the other changes
        # to find out what has lowered performance.
        # Issues are now: conv-consensus CEE; lower accuray of GPPL and crowdGPPL for all conv tests; lower accuracy on sushi B.

        # # # this doesn't really make sense -- the 2 is added on inside pref_likelihood again
        # f_samples = np.random.normal(loc=0, scale=np.sqrt(2),
        #                 size=(self.N, 1000))
        #
        # phi = pref_likelihood(f_samples, v=self.tpref_v, u=self.tpref_u)
        # v_prior = np.var(phi)
        #
        # # the below is a very rough approximation because it ignores the covariance between points,
        # # hence m_prior values will be more extreme and v_prior likely larger.
        # # In practice this seemed to work well in the single user model and is quick to compute.
        # # Assume that each data point has the same prior...
        # a_plus_b = 1.0 / (v_prior / (m_prior*(1-m_prior))) - 1
        # a = a_plus_b * m_prior
        # b = a_plus_b * (1 - m_prior)
        #
        # nu0 = np.array([a, b])
        # # Noise in observations
        # nu0_total = np.sum(nu0, axis=0)
        # obs_mean = (self.z + nu0[1]) / (1 + nu0_total)
        # #var_obs_mean = obs_mean * (1 - obs_mean) / (1 + nu0_total + 1)  # uncertainty in obs_mean
        # Q3 = (obs_mean * (1 - obs_mean))
        # Q3 = Q3.flatten()

        return Q

    def _init_w(self):
        # initialise the factors randomly -- otherwise they can get stuck because there is nothing to differentiate them
        # i.e. the cluster identifiability problem
        # self.w = np.zeros((self.N, self.Nfactors))
        self.w = self.K_nm.dot(self.invK_mm).dot(self.w_u)
        # save for later
        batchsize = 500
        nbatches = int(np.ceil(self.N / float(batchsize) ))

        if self.N > self.max_Kw_size:
            self.Kw_file_tag = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            if not os.path.exists('./tmp'):
                os.mkdir('./tmp')
            self.Kw = np.memmap('./tmp/Kw_%s.tmp' % self.Kw_file_tag, dtype=float, mode='w+', shape=(self.N, self.N))
        else:
            self.Kw = np.zeros((self.N, self.N))

        # # Don't compute all the combinations -- save this until each iteration requires them
        # # so that pairs that are never compared are ignored.
        # for b in range(nbatches):
        #
        #     logging.debug('Computing Kw batch %i' % b)
        #
        #     end1 = (b+1)*batchsize
        #     if end1 > self.N:
        #         end1 = self.N
        #
        #     for b2 in range(nbatches):
        #
        #         end2 = (b2+1)*batchsize
        #         if end2 > self.N:
        #             end2 = self.N
        #
        #         self.Kw[b*batchsize:(b+1)*batchsize, :][:, b2*batchsize:(b2+1)*batchsize] = self.kernel_func(
        #             self.obs_coords[b*batchsize:end1, :], self.ls, self.obs_coords[b2*batchsize:end2, :])

        self.Kw[range(self.N), range(self.N)] = 1.0

        if self.N > self.max_Kw_size:
            self.Kw.flush()


        if not self.new_obs:
            return

        self.Q = self._estimate_obs_noise()

    def _get_Kw(self, idxs0=None, idxs1=None):

        if idxs0 is None or idxs1 is None:
            u = self.tpref_u[self.data_obs_idx_i]
            v = self.tpref_v[self.data_obs_idx_i]
        else:
            u = idxs1
            v = idxs0

        uninited_idxs = self.Kw[u, v] == 0
        if np.any(uninited_idxs):
            u_new = u[uninited_idxs]
            v_new = v[uninited_idxs]

            Kw_pairs = self.kernel_func(self.obs_coords[u_new, :], self.ls,
                                        self.obs_coords[v_new, :], vector=True).flatten()

            self.Kw[u_new, v_new] = Kw_pairs
            self.Kw[v_new, u_new] = Kw_pairs

        if self.N > self.max_Kw_size: # flush any memmapped objects
            self.Kw.flush()

        return self.Kw

    def _init_t(self):
        self.t = np.zeros((self.N, 1))
        self.st = self.shape_st0 / self.rate_st0

        if not self.use_t:
            return

    def _init_y(self):
        self.y_var = np.ones((self.Nfactors, self.Npeople))

        if self.person_features is None:
            self.y = self.y_u
            self.wy = self._wy()

            return
        else:
            self.y = np.zeros((self.Nfactors, self.Npeople))
            for f in range(self.Nfactors):
                if f in self.factors_with_features:
                    self.y[f] = self.Ky_nm.dot(self.invKy_mm).dot(self.y_u[f])
                else:
                    self.y[f] = self.y_u[f]

        self.wy = self._wy()


    def _wy(self, w= None, y=None, peeps=None):

        if w is None:
            w = self.w
        if y is None:
            y = self.y

        if not self.personal_component:
            return w.dot(y)

        wy = w[:, :self.Nfactors-self.Npeople].dot(y[:self.Nfactors-self.Npeople, :])

        if peeps is None:
            wy += self.w[:, -self.Npeople:]
        else:
            wy[:, peeps] += self.w[:, peeps + self.Nfactors - self.Npeople]

        # peeps = np.zeros((self.N, self.Npeople))
        # for n in range(self.N):
        #     uidxs = (self.tpref_v == n) | (self.pref_u == n)
        #     uidxs = self.personIDs[uidxs]
        #     uidxs = np.unique(uidxs)
        #
        #     peeps[n, uidxs] = w[n:n+1, uidxs]
        #
        # wy += peeps
        return wy

    def _init_params(self):
        if self.Nfactors is None or self.Npeople < self.Nfactors:  # not enough items or people
            self.Nfactors = self.Npeople

        self._init_covariance()

        # initialise the inducing points first
        self._choose_inducing_points()

        self.ls = np.zeros(self.nitem_features) + self.ls

        self.G = 0

        self._init_w()
        self._init_y()
        self._init_t()


    def _compute_jacobian(self, obs_f=None):
        if obs_f is None:
            self.obs_f = (self.wy + self.t)
            self.obs_f_flat = self.obs_f.T.reshape(self.N * self.Npeople, 1)
            obs_f = self.obs_f_flat

        phi, g_mean_f = pref_likelihood(obs_f, v=self.pref_v, u=self.pref_u, return_g_f=True)  # first order Taylor series approximation
        J = 1 / (2 * np.pi) ** 0.5 * np.exp(-g_mean_f ** 2 / 2.0) * np.sqrt(0.5)
        J = J[self.data_obs_idx_i, :]

        s = (self.pref_v[self.data_obs_idx_i, None] == self.joint_idx_i[None, :]).astype(int) - \
            (self.pref_u[self.data_obs_idx_i, None] == self.joint_idx_i[None, :]).astype(int)

        J_df = J * s

        return J_df


    def _expec_t(self, update_s=True, update_G=True):

        if not update_G:
            max_iter_G = 1
        else:
            max_iter_G = self.max_iter_G
            self._update_sample() # only update the sample during normal iterations, not when G has been fixed

        if not self.use_t:
            return

        if update_G:
            self.prev_tinvS = self.tinvS
            self.prev_tinvSm = self.tinvSm

        N = self.ninducing

        rho_i = (self.vb_iter + self.delay) ** (-self.forgetting_rate)
        w_i = self.nobs / float(self.update_size)

        # Does w_idx_i contain duplicates? If so, these might be double counting?
        covpair = self.invK_mm.dot(self.K_nm[self.w_idx_i].T)
    
        G_update_rate = 1.0
        diff = 0

        for G_iter in range(max_iter_G):

            oldG = self.G
            if update_G:
                self.G = self.G * (1-G_update_rate) + self._compute_jacobian() * G_update_rate

            # we need to map from the real Npeople points to the inducing points.
            KKG = covpair.dot(self.G.T)
            self.Sigma_t = (KKG/self.Q[None, self.data_obs_idx_i]).dot(KKG.T)

            # need to get invS for current iteration and merge using SVI weighted sum
            self.tinvS = (1-rho_i) * self.prev_tinvS + rho_i * (self.invK_mm*self.shape_st/self.rate_st + w_i * self.Sigma_t)

            z0 = pref_likelihood(self.obs_f_flat, v=self.pref_v[self.data_obs_idx_i], u=self.pref_u[self.data_obs_idx_i]) \
                 - self.G.dot(self.t[self.w_idx_i, :])# P x NU_i the latent factors cancel!

            invQ_f = (self.G.T / self.Q[None, self.data_obs_idx_i]).dot(self.z[self.data_obs_idx_i] - z0)
            x = covpair.dot(invQ_f)

            # need to get x for current iteration and merge using SVI weighted sum
            self.tinvSm = (1-rho_i) * self.prev_tinvSm + rho_i * w_i * x

            self.tS = np.linalg.inv(self.tinvS)
            self.t_u = self.tS.dot(self.tinvSm)

            prev_diff_G = diff  # save last iteration's difference
            diff = np.max(np.abs(oldG - self.G))

            if np.abs(np.abs(diff) - np.abs(prev_diff_G)) < 1e-3 and G_update_rate > 0.1:
                G_update_rate *= 0.9

            if self.verbose:
                logging.debug("expec_t: iter %i, G-diff=%f" % (G_iter, diff))

            if diff < self.conv_threshold_G:
                break

            self.t, _ = inducing_to_observation_moments(self.K_mm * self.rate_st / self.shape_st,
                                                    self.invK_mm, self.K_nm, self.t_u, self.t_mu0)

        if update_s:
            self.shape_st, self.rate_st = expec_output_scale(self.shape_st0, self.rate_st0, N,
                                                         self.invK_mm, self.t_u, np.zeros((N, 1)),
                                                         f_cov=self.tS)
            self.st = self.shape_st / self.rate_st


    def _expec_w(self, update_s=True, update_G=True):
        """
        Compute the expectation over the latent features of the items and the latent personality components
        """

        if not update_G:
            max_iter_G = 1
        else:
            max_iter_G = self.max_iter_G

        self.prev_winvS = self.winvS.copy()
        self.prev_winvSm = self.winvSm.copy()

        rho_i = (self.vb_iter + self.delay) ** (-self.forgetting_rate)
        w_i = self.nobs / float(self.update_size)

        covpair = self.invK_mm.dot(self.K_nm[self.w_idx_i].T)

        G_update_rate = 1.0
        diff = 0

        for G_iter in range(max_iter_G):

            oldG = self.G
            if update_G:
                self.G = self.G * (1-G_update_rate)  + self._compute_jacobian() * G_update_rate

            # we need to map from the real Npeople points to the inducing points.
            for f in range(self.Nfactors):

                if self.personal_component and f >= self.Nfactors-self.Npeople:

                    user = f - self.Nfactors + self.Npeople
                    user_idxs = self.y_idx_i == user # only update the data points corresponding to the
                    # user who owns this factor
                    if np.sum(user_idxs) == 0:
                        continue

                    if (np.mod(user, 100) == 0):
                        logging.debug('Update personal factor %i' % user)

                    covpair_ones = np.eye(self.N)[:, self.w_idx_i[user_idxs]]

                    user_obs = self.personIDs[self.data_obs_idx_i] == user
                    G = self.G[user_obs, :][:, self.w_idx_i[user_idxs]]
                    Sigma_w_f = covpair_ones.dot(G.T / self.Q[None, self.data_obs_idx_i[user_obs]]).dot(G).dot(covpair_ones.T)

                    # need to get invS for current iteration and merge using SVI weighted sum
                    self.winvS[f] = (1-rho_i) * self.prev_winvS[f] + rho_i * (self.invKv*self.shape_sw[f]
                                / self.rate_sw[f] + w_i * Sigma_w_f)

                    z0 = pref_likelihood(self.obs_f_flat, v=self.pref_v[self.data_obs_idx_i[user_obs]],
                                         u=self.pref_u[self.data_obs_idx_i[user_obs]]) \
                         - G.dot(self.w[self.w_idx_i[user_idxs], f:f + 1])  # P x NU_i

                    invQ_f = (G.T / self.Q[None, self.data_obs_idx_i[user_obs]]).dot(self.z[self.data_obs_idx_i[user_obs]] - z0)

                    x = covpair_ones.dot(invQ_f)
                else:
                    # scale the precision by y
                    # scaling_f = self.y[f:f+1, self.y_idx_i].T.dot(self.y[f:f+1, self.y_idx_i]) + \
                    #             self.y_cov_i[f][self.uy_idx_i, :][:, self.uy_idx_i]
                    scaling_f = (self.y[f, self.personIDs[self.data_obs_idx_i]] ** 2
                                 + self.y_cov_i[f][self.personIDs[self.data_obs_idx_i]])[None, :]

                    #Sigma_w_f = covpair.dot(invQGT.dot(self.G) * scaling_f).dot(covpair.T)
                    KKG = covpair.dot(self.G.T)
                    Sigma_w_f = (KKG * (scaling_f/self.Q[None, self.data_obs_idx_i])).dot(KKG.T)

                    # need to get invS for current iteration and merge using SVI weighted sum
                    self.winvS[f] = (1-rho_i) * self.prev_winvS[f] + rho_i * (self.invK_mm*self.shape_sw[f]
                                / self.rate_sw[f] + w_i * Sigma_w_f)

                    z0 = pref_likelihood(self.obs_f_flat, v=self.pref_v[self.data_obs_idx_i], u=self.pref_u[self.data_obs_idx_i]) \
                         - self.G.dot(self.w[self.w_idx_i, f:f+1] * self.y[f:f+1, self.y_idx_i].T) # P x NU_i

                    invQ_f = (self.y[f:f+1, self.y_idx_i].T * self.G.T / self.Q[None, self.data_obs_idx_i]).dot(
                        self.z[self.data_obs_idx_i] - z0)

                    x = covpair.dot(invQ_f)

                # need to get x for current iteration and merge using SVI weighted sum
                self.winvSm[:, f] = (1-rho_i) * self.prev_winvSm[:, f] + rho_i * w_i * x.flatten()

                self.wS[f] = np.linalg.inv(self.winvS[f])
                self.w_u[:, f] = self.wS[f].dot(self.winvSm[:, f])

                if self.personal_component and f >= self.Nfactors-self.Npeople:
                    K_mm = self.Kv
                    K_nm = self.Kv
                    invK_mm = self.invKv
                    self.obs_f[self.uw_i, f-self.Nfactors+self.Npeople] -= self.w[self.uw_i, f]

                else:
                    K_mm = self.K_mm
                    K_nm = self.K_nm
                    invK_mm = self.invK_mm
                    self.obs_f[self.uw_i, :] -= self.w[self.uw_i, f:f+1].dot(self.y[f:f+1])


                self.w[:, f:f+1], _ = inducing_to_observation_moments(K_mm / self.shape_sw[f] * self.rate_sw[f],
                                    invK_mm, K_nm, self.w_u[:, f:f+1], 0)

                if self.personal_component and f >= self.Nfactors-self.Npeople:
                    self.obs_f[self.uw_i, f-self.Nfactors+self.Npeople] += self.w[self.uw_i, f]
                else:
                    self.obs_f[self.uw_i, :] += self.w[self.uw_i, f:f+1].dot(self.y[f:f+1])

                self.obs_f_flat = self.obs_f.T.reshape(self.N * self.Npeople, 1)

            prev_diff_G = diff  # save last iteration's diffhttps://www.theguardian.com/uk/commentisfreeerence
            diff = np.max(np.abs(oldG - self.G))

            if np.abs(np.abs(diff) - np.abs(prev_diff_G)) < 1e-3 and G_update_rate > 0.1:
                G_update_rate *= 0.9

            if self.verbose:
                logging.debug("expec_w: iter %i, G-diff=%f" % (G_iter, diff))

            if diff < self.conv_threshold_G:
                break

            if update_G:
                self.wy = self.w.dot(self.y)

        if self.verbose:
            logging.debug('Computing w_cov_i')
        Kw_i = self._get_Kw()[self.uw_i, :][:, self.uw_i]
        K_nm_i = self.K_nm[self.uw_i]

        self.w_cov_i = np.zeros((self.Nfactors, self.uw_i.shape[0], self.uw_i.shape[0]))

        covpair = K_nm_i.dot(self.invK_mm)
        sw = self.shape_sw / self.rate_sw

        for f in range(self.Nfactors):

            if self.personal_component and f >= self.Nfactors - self.Npeople:
                break

            self.w_cov_i[f] = Kw_i / sw[f] + covpair.dot(self.wS[f] - self.K_mm/sw[f]).dot(covpair.T)

            if not update_s:
                continue

            self.shape_sw[f], self.rate_sw[f] = expec_output_scale(self.shape_sw0, self.rate_sw0, self.ninducing,
               self.invK_mm, self.w_u[:, f:f + 1], np.zeros((self.ninducing, 1)), f_cov=self.wS[f])


    def _expec_y(self, update_s=True, update_G=True):

        if not update_G:
            max_iter_G = 1
        else:
            max_iter_G = self.max_iter_G

        self.prev_yinvSm = self.yinvSm.copy()
        self.prev_yinvS = self.yinvS.copy()

        rho_i = (self.vb_iter + self.delay) ** (-self.forgetting_rate)
        w_i = np.sum(self.nobs) / float(self.update_size)

        if self.person_features is not None:
            covpair = self.invKy_mm.dot(self.Ky_nm[self.y_idx_i].T)

            if self.factors_with_features is not None:
                covpair_ones = np.eye(self.Npeople)[self.y_idx_i, :].T

        else:
            covpair_ones = np.eye(self.Npeople)[self.y_idx_i, :].T

        if self.verbose:
            logging.debug('_expec_y: starting update.')

        G_update_rate = 1.0
        diff = 0

        for G_iter in range(max_iter_G):
            oldG = self.G

            if update_G:
                self.G = self.G * (1-G_update_rate) + self._compute_jacobian() * G_update_rate

            invQG =  self.G.T / self.Q[None, self.data_obs_idx_i]

            for f in range(self.Nfactors):

                if self.personal_component and f >= self.Nfactors-self.Npeople:
                    break

                scaling_f = self.w[self.w_idx_i, f:f+1].dot(self.w[self.w_idx_i, f:f+1].T) \
                            + self.w_cov_i[f][self.uw_idx_i, :][:, self.uw_idx_i]

                z0 = pref_likelihood(self.obs_f_flat, v=self.pref_v[self.data_obs_idx_i], u=self.pref_u[self.data_obs_idx_i]) \
                     - self.G.dot(self.w[self.w_idx_i, f:f+1] * self.y[f:f+1, self.y_idx_i].T)  # P x NU_i

                invQ_f = (self.w[self.w_idx_i, f:f+1] * self.G.T / self.Q[None, self.data_obs_idx_i]).dot(
                    self.z[self.data_obs_idx_i] - z0)

                self.obs_f[:, self.uy_i] -= self.w[:, f:f+1].dot(self.y[f:f+1, self.uy_i])

                # need to get invS for current iteration and merge using SVI weighted sum
                if self.person_features is not None and f in self.factors_with_features:
                    Sigma_y_f = covpair.dot(scaling_f * invQG.dot(self.G)).dot(covpair.T)

                    self.yinvS[f] = (1-rho_i) * self.prev_yinvS[f] + rho_i * (self.shape_sy[f] / self.rate_sy[f] *
                                                                              self.invKy_mm + w_i * Sigma_y_f)

                    x = covpair.dot(invQ_f)

                    self.yinvSm[f] = (1-rho_i) * self.prev_yinvSm[f] + rho_i * w_i * x.flatten()

                    self.yS[f] = np.linalg.inv(self.yinvS[f])
                    self.y_u[f] = self.yS[f].dot(self.yinvSm[f])

                    yf, _ = inducing_to_observation_moments(self.Ky_mm / self.shape_sy[f] * self.rate_sy[f],
                                                            self.invKy_mm, self.Ky_nm, self.y_u[f][:, None], 0)

                    self.y[f:f + 1] = yf.T
                else:
                    Sigma_y_f = covpair_ones.dot(scaling_f * invQG.dot(self.G)).dot(covpair_ones.T)

                    Sigma_y_f = np.diag(Sigma_y_f)
                    self.yinvS[f] = (1-rho_i) * self.prev_yinvS[f] + rho_i * (self.shape_sy[f] / self.rate_sy[f]
                                                                                + w_i * Sigma_y_f)

                    x = covpair_ones.dot(invQ_f)

                    self.yinvSm[f] = (1-rho_i) * self.prev_yinvSm[f] + rho_i * w_i * x.flatten()

                    self.yS[f] = 1.0 / self.yinvS[f]
                    self.y_u[f] = (self.yS[f].T * self.yinvSm[f]).T
                    self.y[f] = self.y_u[f]

                self.obs_f[:, self.uy_i] += self.w[:, f:f+1].dot(self.y[f:f+1, self.uy_i])
                self.obs_f_flat = self.obs_f.T.reshape(self.N * self.Npeople, 1)

            prev_diff_G = diff  # save last iteration's difference

            diff = np.max(np.abs(oldG - self.G))

            if np.abs(np.abs(diff) - np.abs(prev_diff_G)) < 1e-3 and G_update_rate > 0.1:
                G_update_rate *= 0.9

            if self.verbose:
                logging.debug("expec_y: iter %i, G-diff=%f" % (G_iter, diff))

            if diff < self.conv_threshold_G:
                break

            if update_G:
                self.wy = self.w.dot(self.y)

        for f in range(self.Nfactors):

            if self.personal_component and f >= self.Nfactors-self.Npeople:
                break

            if self.person_features is not None and f in self.factors_with_features:
                self.shape_sy[f], self.rate_sy[f] = expec_output_scale(self.shape_sy0, self.rate_sy0, self.y_ninducing,
                   self.invKy_mm, self.y_u[f][None, :], np.zeros((self.y_ninducing, 1)), f_cov=self.yS[f])
            else:
                self.shape_sy[f], self.rate_sy[f] = expec_output_scale(self.shape_sy0, self.rate_sy0, self.Npeople,
                   np.ones(self.Npeople), self.y_u[f][None, :], np.zeros((self.Npeople, 1)), f_cov=self.yS[f])

        self.wy = self._wy()

    def _update_sample_idxs(self, data_obs_idx_i=None, compute_y_var=True):
        # do this in first iteration
        if self.nsplits == 0:
            self.nsplits = int(np.ceil(self.nobs / float(self.update_size)))

            if self.exhaustive_train:
                if self.min_iter < self.nsplits:
                    self.min_iter = self.nsplits
                if self.max_iter < self.min_iter * self.exhaustive_train:
                    self.max_iter = self.min_iter * self.exhaustive_train

        if data_obs_idx_i is None:
            if self.data_splits is None or np.mod(self.current_data_split+1, self.nsplits) == 0:
                # create nsplits random splits -- shuffle data and split

                # ensure we sample from each person as soon as possible
                people, first_occs = np.unique(self.personIDs, return_index=True)

                rand_order = np.random.permutation(self.nobs)
                rand_order = rand_order[np.invert(np.in1d(rand_order, first_occs))]
                rand_order = np.concatenate((first_occs, rand_order))

                self.data_splits = []

                for n in range(self.nsplits):
                    ending = self.update_size * (n + 1)
                    if ending > self.nobs:
                        ending = self.nobs
                    self.data_splits.append(rand_order[self.update_size*n:ending])

                self.current_data_split = 0
            else:
                self.current_data_split += 1

            self.data_obs_idx_i = self.data_splits[self.current_data_split]
            # self.data_obs_idx_i = np.sort(np.random.choice(self.nobs, self.update_size, replace=False))
        else:
            self.data_obs_idx_i = data_obs_idx_i

        data_idx_i = np.zeros((self.N, self.Npeople), dtype=bool)
        data_idx_i[self.tpref_v[self.data_obs_idx_i], self.personIDs[self.data_obs_idx_i]] = True
        data_idx_i[self.tpref_u[self.data_obs_idx_i], self.personIDs[self.data_obs_idx_i]] = True

        separate_idx_i = np.argwhere(data_idx_i)
        self.w_idx_i = separate_idx_i[:, 0]
        self.y_idx_i = separate_idx_i[:, 1]
        self.joint_idx_i = self.w_idx_i + (self.N * self.y_idx_i)

        self.uw_i, self.uw_idx_i = np.unique(self.w_idx_i, return_inverse=True)
        self.uy_i, self.uy_idx_i = np.unique(self.y_idx_i, return_inverse=True)

        if not compute_y_var:
            return

        if self.verbose:
            logging.debug('Computing y_cov_i')

        if self.person_features is not None:
            self.y_cov_i = np.zeros((self.Nfactors, self.Npeople))

            covpair = self.Ky_nm.dot(self.invKy_mm)
            sy = self.shape_sy / self.rate_sy

            for f in range(self.Nfactors):
                if f in self.factors_with_features:
                    self.y_cov_i[f] = 1.0 / sy[f] + np.sum(covpair.dot(self.yS[f] - self.Ky_mm/sy[f]) * covpair, axis=1)
                else:
                    self.y_cov_i[f] = self.yS[f]

        else:
            self.y_cov_i = self.yS

    def _update_sample(self):
        self._update_sample_idxs()
        self.G = 0 # need to reset G because we have a new sample to compute it for

    def data_ll(self, logrho, lognotrho):
        bc = binom(np.ones(self.z.shape), self.z)
        logbc = np.log(bc)
        lpobs = np.sum(self.z * logrho + (1 - self.z) * lognotrho)
        lpobs += np.sum(logbc)

        data_ll = lpobs
        return data_ll

    def _logpD(self):
        # this is possible because the probit likelihood can be split into a discrete and a Gaussian noise component.
        # The Gaussian noise component has log p = 0 so is ignored. The noise component ends up depending only on the
        # mean of the latent function and works out to this -- see notes on variational probit regression here:
        # https://rpubs.com/cakapourani/variational-bayes-bpr
        rho = pref_likelihood(self.obs_f_flat, v=self.pref_v, u=self.pref_u)
        rho = temper_extreme_probs(rho)
        data_ll = self.data_ll(np.log(rho), np.log(1 - rho))

        return data_ll

    def lowerbound(self):

        data_ll = self._logpD()

        Elnsw = psi(self.shape_sw) - np.log(self.rate_sw)
        Elnsy = psi(self.shape_sy) - np.log(self.rate_sy)
        if self.use_t:
            Elnst = psi(self.shape_st) - np.log(self.rate_st)
            st = self.st
        else:
            Elnst = 0
            st = 1

        sw = self.shape_sw / self.rate_sw
        sy = self.shape_sy / self.rate_sy

        # the parameter N is not multiplied here by Nfactors because it will be multiplied by the s value for each
        # factor and summed inside the function
        logpw = np.sum([expec_pdf_gaussian(self.K_mm, self.invK_mm, Elnsw[f], self.ninducing, sw[f],
                                           self.w_u[:, f:f+1], 0, self.wS[f], 0)
                        for f in range(self.Nfactors)])

        logqw = np.sum([expec_q_gaussian(self.wS[f], self.ninducing * self.Nfactors) for f in range(self.Nfactors)])

        if self.use_t:
            logpt = expec_pdf_gaussian(self.K_mm, self.invK_mm, Elnst, self.ninducing, st, self.t_u, self.t_mu0,
                                       0, 0) - 0.5 * self.ninducing
            logqt = expec_q_gaussian(self.tS, self.ninducing)
        else:
            logpt = 0
            logqt = 0

        if self.person_features is not None:
            logpy = np.sum([expec_pdf_gaussian(self.Ky_mm if f in self.factors_with_features else np.ones(self.Npeople),
                                           self.invKy_mm if f in self.factors_with_features else np.ones(self.Npeople),
                                           0, self.y_ninducing, sy[f], self.y_u[f][:, None], 0, self.yS[f], 0)
                        for f in range(self.Nfactors)])
            logqy = np.sum([expec_q_gaussian(self.yS[f], (self.y_ninducing if f in self.factors_with_features
                                  else self.Npeople) * self.Nfactors) for f in range(self.Nfactors)])
        else:
            logpy = np.sum([expec_pdf_gaussian(self.Ky_mm, self.invKy_mm,
                                           0, self.y_ninducing, sy[f], self.y_u[f][:, None], 0, self.yS[f], 0)
                        for f in range(self.Nfactors if not self.personal_component else (self.Nfactors - self.Npeople))])
            logqy = np.sum([expec_q_gaussian(self.yS[f], self.y_ninducing * self.Nfactors) for f in
                            range(self.Nfactors if not self.personal_component else (self.Nfactors - self.Npeople))])


        logps_w = 0
        logqs_w = 0

        logps_y = 0
        logqs_y = 0

        for f in range(self.Nfactors):
            logps_w += lnp_output_scale(self.shape_sw0, self.rate_sw0, self.shape_sw[f], self.rate_sw[f], sw[f],
                                        Elnsw[f])
            logqs_w += lnq_output_scale(self.shape_sw[f], self.rate_sw[f], sw[f], Elnsw[f])

            logps_w += lnp_output_scale(self.shape_sy0, self.rate_sy0, self.shape_sy[f], self.rate_sy[f], sy[f],
                                        Elnsy[f])
            logqs_w += lnq_output_scale(self.shape_sy[f], self.rate_sy[f], sy[f], Elnsy[f])

        logps_t = lnp_output_scale(self.shape_st0, self.rate_st0, self.shape_st, self.rate_st, st, Elnst)
        logqs_t = lnq_output_scale(self.shape_st, self.rate_st, st, Elnst)

        w_terms = logpw - logqw + logps_w - logqs_w
        y_terms = logpy - logqy + logps_y - logqs_y
        t_terms = logpt - logqt + logps_t - logqs_t

        lb = data_ll + t_terms + w_terms + y_terms

        if self.verbose:
            logging.debug('s_w=%s' % (sw))
            logging.debug('s_y=%s' % (sy))
            logging.debug('s_t=%f' % (st))

        if self.verbose:
            logging.debug('likelihood=%.3f, wterms=%.3f, yterms=%.3f, tterms=%.3f' % (data_ll, w_terms, y_terms, t_terms))
            logging.debug("Iteration %i: Lower bound = %.3f, " % (self.vb_iter, lb))
            logging.debug("t: %.2f, %.2f" % (np.min(self.t), np.max(self.t)))
            logging.debug("w: %.2f, %.2f" % (np.min(self.w), np.max(self.w)))
            logging.debug("y: %f, %f" % (np.min(self.y), np.max(self.y)))

        return lb

    def _compute_cov_w(self, cov_0, cov_1):

        Kw = self._get_Kw(cov_0, cov_1)
        N = Kw.shape[0]

        cov_w = np.zeros((self.Nfactors, N, N))
        covpair = self.K_nm.dot(self.invK_mm)

        for f in range(self.Nfactors):
            cov_w[f] = Kw * self.rate_sw[f] / self.shape_sw[f] + \
                       covpair.dot(self.wS[f] - self.K_mm * self.rate_sw[f] / self.shape_sw[f]).dot(covpair.T)

        return cov_w

    def _compute_cov_t(self, cov_0, cov_1):
        if self.use_t:
            covpair = self.K_nm.dot(self.invK_mm)

            covpair_uS = covpair.dot(self.tS)
            cov_t = self._get_Kw(cov_0, cov_1) * self.rate_st / self.shape_st + (covpair_uS -
                         covpair.dot(self.K_mm * self.rate_st / self.shape_st)).dot(covpair.T)
            return cov_t
        else:
            return None

    def _predict_w_t(self, coords_1, return_cov=True):

        # kernel between pidxs and t
        if self.verbose:
            logging.debug('Computing K_nm in predict_w_t')
        K = self.kernel_func(coords_1, self.ls, self.inducing_coords)
        if self.verbose:
            logging.debug('Computing K_nn in predict_w_t')
        K_starstar = self.kernel_func(coords_1, self.ls, coords_1)
        covpair = K.dot(self.invK_mm)
        N = coords_1.shape[0]

        # use kernel to compute t.
        if self.use_t:
            t_out = K.dot(self.invK_mm).dot(self.t_u)

            covpair_uS = covpair.dot(self.tS)
            if return_cov:
                cov_t = K_starstar * self.rate_st / self.shape_st + (covpair_uS -
                             covpair.dot(self.K_mm * self.rate_st / self.shape_st)).dot(covpair.T)
            else:
                cov_t = None
        else:
            t_out = np.zeros((N, 1))
            if return_cov:
                cov_t = np.zeros((N, N))
            else:
                cov_t = None

        # kernel between pidxs and w -- use kernel to compute w. Don't need Kw_mm block-diagonal matrix
        w_out = K.dot(self.invK_mm).dot(self.w_u)

        if return_cov:
            cov_w = np.zeros((self.Nfactors, N, N))
            for f in range(self.Nfactors):
                cov_w[f] = K_starstar  * self.rate_sw[f] / self.shape_sw[f] + \
                                covpair.dot(self.wS[f] - self.K_mm * self.rate_sw[f] / self.shape_sw[f]).dot(covpair.T)
        else:
            cov_w = None

        return t_out, w_out, cov_t, cov_w

    def predict_t(self, item_features=None):
        '''
        Predict the common consensus function values using t
        '''
        if item_features is None:
            # reuse the training points
            t = self.t
        else:
            # use kernel to compute t.
            if self.use_t:
                # kernel between pidxs and t
                if self.verbose:
                    logging.debug('Computing K_nm in predict_t')
                K = self.kernel_func(item_features, self.ls, self.inducing_coords)
                t = K.dot(self.invK_mm).dot(self.t_u)
            else:
                N = item_features.shape[0]
                t = np.zeros((N, 1))

        return t

    def predict_common(self, item_features, item_0_idxs, item_1_idxs):
        '''
        Predict the common consensus pairwise labels using t.
        '''
        if not self.use_t:
            return np.zeros(len(item_0_idxs))

        # TODO Replace the pre-calculation of Kw with a function that computes on demand
        #  (for any pairs that are zero) and saves results. Do this so we can generate a better plot for
        #  figure 2b.
        if item_features is None:
            K = self.K_nm
            K_starstar = self._get_Kw(item_0_idxs, item_1_idxs)
        else:
            if self.verbose:
                logging.debug('Computing K_nm in predict_common')
            K = self.kernel_func(item_features, self.ls, self.inducing_coords)

            if self.verbose:
                logging.debug('Computing K_nn in predict_common')
            K_starstar = self.kernel_func(item_features, self.ls, item_features)

        covpair = K.dot(self.invK_mm)
        covpair_uS = covpair.dot(self.tS)

        t_out = K.dot(self.invK_mm).dot(self.t_u)
        cov_t = K_starstar * self.rate_st / self.shape_st + (covpair_uS
                    - covpair.dot(self.K_mm * self.rate_st / self.shape_st)).dot(covpair.T)

        predicted_prefs = pref_likelihood(t_out, cov_t[item_0_idxs, item_0_idxs]
                                          + cov_t[item_1_idxs, item_1_idxs]
                                          - cov_t[item_0_idxs, item_1_idxs]
                                          - cov_t[item_1_idxs, item_0_idxs],
                                          subset_idxs=[], v=item_0_idxs, u=item_1_idxs)
        predicted_prefs = temper_extreme_probs(predicted_prefs)

        return predicted_prefs

    def _y_var(self):
        if self.person_features is None:
            return self.yS

        v = np.array([inducing_to_observation_moments(self.Ky_mm / self.shape_sy[f] * self.rate_sy[f],
                          self.invKy_mm, self.Ky_nm, self.y_u[f][:, None], 0, S=self.yS[f],
                          Ks_nn=self.rate_sy[f] / self.shape_sy[f])[1]
                      if f in self.factors_with_features else self.yS[f]
                      for f in range(self.Nfactors)])
        return v

    def _predict_y_tr(self, return_cov):

        y_out, y_var = super(CollabPrefLearningSVI, self)._predict_y_tr(return_cov)

        if self.y_ninducing == self.Npeople or not self.use_local_obs_posterior_y:
            return y_out, y_var

        # batch up the pairwise labels
        P = self.pref_u.shape[0]
        nbatches = int(np.ceil(P / float(self.update_size)))

        prec_out = []
        prec_out_y_out = []

        for f in range(self.Nfactors):
            if y_var is not None:
                var_yf_given_u = y_var[f]
            else:
                covpair = self.Ky_nm.dot(self.invKy_mm)
                var_yf_given_u = self.rate_sy[f] / self.shape_sy[f] + np.diag(covpair.dot(self.yS[f] -
                                          self.Ky_mm / self.shape_sy[f] * self.rate_sy[f]).dot(covpair.T))

            prec_out.append(1.0 / var_yf_given_u)
            prec_out_y_out.append( y_out[f] / var_yf_given_u)

        Q = self.Q

        obs_f = (self.w.dot(y_out) + self.t).T.reshape(self.N * self.Npeople, 1)

        # test how likely the observations are given current predictions.
        # The calculation ignores all the other factors!
        for b in range(nbatches):
            # check whether having only one batch helps.
            b_end = (b+1) * self.update_size
            if b_end > P:
                b_end = P

            data_obs_idx_i = np.arange(b * self.update_size, b_end)
            self._update_sample_idxs(data_obs_idx_i, False)

            G = self._compute_jacobian(obs_f)
            # equivalent to the covpair
            matcher = np.zeros((self.Npeople, G.shape[1]))
            matcher[self.y_idx_i, np.arange(matcher.shape[1])] = 1

            self.w_cov_i = np.zeros((self.Nfactors, self.uw_i.shape[0], self.uw_i.shape[0]))

            K_nm_i = self.K_nm[self.uw_i]
            covpair = K_nm_i.dot(self.invK_mm)
            sw = self.shape_sw / self.rate_sw
            Kw_i = self._get_Kw()[self.uw_i, :][:, self.uw_i]

            for f in range(self.Nfactors):
                self.w_cov_i[f] = Kw_i / sw[f] + covpair.dot(self.wS[f] - self.K_mm / sw[f]).dot(covpair.T)
                scaling_f = self.w[self.w_idx_i, f:f+1].dot(self.w[self.w_idx_i, f:f+1].T) \
                            + self.w_cov_i[f][self.uw_idx_i, :][:, self.uw_idx_i]

                GinvQ = G.T / Q[None, self.data_obs_idx_i]
                obs_prec = np.diag(matcher.dot(scaling_f * GinvQ.dot(G)).dot(matcher.T))

                prec_out[f] += obs_prec

                z0 = pref_likelihood(obs_f, subset_idxs=self.data_obs_idx_i, v=self.pref_v, u=self.pref_v) \
                     - G.dot(self.w[self.w_idx_i, f:f+1] * self.y[f:f+1, self.y_idx_i].T)
                y = self.z[self.data_obs_idx_i] - z0

                prec_out_y_out[f] += matcher.dot(self.w[self.w_idx_i, f:f+1] * GinvQ).dot(y).flatten()

        y_var = []
        for f in range(self.Nfactors):
            y_var.append(1.0 / prec_out[f])
            y_out[f] = prec_out_y_out[f] / prec_out[f]

        return y_out, np.array(y_var)

    def _predict_y(self, person_features, return_cov=True):

        if person_features is None and self.person_features is None:

            if return_cov:
                cov_y = np.zeros((self.Nfactors, self.y_ninducing, self.y_ninducing))
                for f in range(self.Nfactors):
                    cov_y[f] = self.yS[f]
            else:
                cov_y = None

            return self.y_u, cov_y

        elif person_features is None:
            # we have a person that we have seen before
            Ky = self.Ky_nm
            Ky_starstar = self.Ky_nm.dot(self.invKy_mm).dot(self.Ky_nm.T)
            Npeople = self.person_features.shape[0]

        else:
            if self.verbose:
                logging.debug('Computing Ky_nm in predict_y')
            Ky = self.y_kernel_func(person_features, self.lsy, self.y_inducing_coords)
            if self.verbose:
                logging.debug('Computing Ky_nn in predict_y')
            Ky_starstar = self.y_kernel_func(person_features, self.lsy, person_features)
            Npeople = person_features.shape[0]

        covpair = Ky.dot(self.invKy_mm)

        y_out = np.zeros((self.Nfactors, Npeople))
        for f in range(self.Nfactors):
            y_out[f] = Ky.dot(self.invKy_mm).dot(self.y_u[f])

        if return_cov:
            cov_y = np.zeros((self.Nfactors, Npeople, Npeople))
            for f in range(self.Nfactors):
                if f in self.factors_with_features:
                    cov_y[f] = Ky_starstar * self.rate_sy[f] / self.shape_sy[f] \
                       + covpair.dot(self.yS[f] - self.Ky_mm / self.shape_sy[f] * self.rate_sy[f]).dot(covpair.T)
                else:
                    cov_y[f] = self.yS[f]
        else:
            cov_y = None

        return y_out, cov_y

    def _compute_gradients_all_dims(self, lstype, dimensions):
        mll_jac = np.zeros(len(dimensions), dtype=float)

        if lstype == 'item' or (lstype == 'both'):
            common_term = np.sum(np.array([(self.w_u[:, f:f+1].dot(self.w_u[:, f:f+1].T) + self.wS[f]).dot(
                self.shape_sw[f] / self.rate_sw[f] * self.invK_mm) - np.eye(self.ninducing)
                for f in range(self.Nfactors)]), axis=0)
            if self.use_t:
                common_term += (self.t_u.dot(self.t_u.T) + self.tS).dot(self.shape_st / self.rate_st * self.invK_mm) \
                               - np.eye(self.ninducing)

            for dim in dimensions[:self.nitem_features]:
                if self.verbose and np.mod(dim, 1000)==0:
                    logging.debug('Computing gradient for %s dimension %i' % (lstype, dim))
                mll_jac[dim] = self._gradient_dim(self.invK_mm, common_term, 'item', dim)

        if (lstype == 'person' or (lstype == 'both')) and self.person_features is not None:
            common_term = np.sum(np.array([(self.y_u[f:f+1].T.dot(self.y_u[f:f+1,:]) + self.yS[f]).dot(
                self.shape_sy[f] / self.rate_sy[f] * self.invKy_mm)
                                           - np.eye(self.y_ninducing) for f in range(self.Nfactors)]), axis=0)

            for dim in dimensions[self.nitem_features:]:
                if self.verbose and np.mod(dim, 1000)==0:
                    logging.debug('Computing gradient for %s dimension %i' % (lstype, dim))
                mll_jac[dim + self.nitem_features] = self._gradient_dim(self.invKy_mm, common_term, 'person', dim)

        return mll_jac

    def _gradient_dim(self, invK_mm, common_term, lstype, dimension):
        # compute the gradient. This should follow the MAP estimate from chu and ghahramani.
        # Terms that don't involve the hyperparameter are zero; implicit dependencies drop out if we only calculate
        # gradient when converged due to the coordinate ascent method.
        if lstype == 'item':
            dKdls = self.K_mm * self.kernel_der(self.inducing_coords, self.ls, dimension)
        elif lstype == 'person' and self.person_features is not None:
            dKdls = self.Ky_mm * self.kernel_der(self.y_inducing_coords, self.lsy, dimension)

        return 0.5 * np.trace(common_term.dot(dKdls).dot(invK_mm))
