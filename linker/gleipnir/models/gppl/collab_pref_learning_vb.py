"""
Preference learning model for identifying relevant input features of items and people, plus finding latent
characteristics of items and people. Can be used to predict preferences or rank items, therefore could be part of
a recommender system. In this case the method uses both collaborative filtering and item-based similarity.

For preference learning, we use a GP as in Chu and Ghahramani. This has been modified to use VB updates to integrate 
into the complete VB framework and allow an SVI adaptation for scalability. 

For finding the latent factors, we modify the model of Archambeau and Bach 2009 to consider correlations between 
items and people using a GP kernel. We assume only one "view" (from their terminology) that best fits the preferences.
Multiple views could be worthwhile if predicting multiple preference functions?
According to A&B 2009, the generative model for probabilistic projection includes several techniques as special cases:
 - diagonal priors on y gives probabilistic factor analysis
 - isotropic priors give probabilistic PCA
 - our model doesn't allow other specific types, but is intended to be used more generally instead
From A&B'09: "PCA suffers from the fact that each principal component is a linear combination of all the original 
variables. It is thus often difficult to interpret the results." A sparse representation would be easier to interpret. 
The aim is to use as few components as are really necessary, and allow priors to determine a trade-off between sparsity
(and interpretability; possibly also avoidance of overfitting/better generalisation) and accuracy of the low-dimensional
representation (avoiding loss). In preference learning, we would like to be able to predict unseen values in a person-item
matrix when we have limited, noisy data. A sparse representation seems applicable as it would avoid overfitting and 
allow personalised noise models to represent complexity so that the shared latent features are more easily interpretable
as common characteristics. 

Our implementation here is similar to the inverse Gamma prior on the weight precision
proposed by A&B'09, but we use a gamma prior that is conjugate to the Gaussian instead. This makes inference simpler
but may have the disadvantage of not enforcing sparseness so strictly -- it is not clear from A&B'09 why they chose
the non-conjugate option. They also use completely independent scale variables for each weight in the weight matrix,
i.e. for each item x factor pair. We have correlations between items with similar features through a kernel function, 
but we also use a common scale parameter for each feature. This induces sparsity over the features, i.e. reduces the
number of features used but means that all items will have an entry for all the important features. It's unclear what
difference this makes -- perhaps features that are weakly supported by small amounts of data for one item will be pushed
to zero by A&B approach, while our approach will allow them to vary more since the feature is important for other items.
The A&B approach may make more sense for representing rare but important features; our approach would not increase their 
scale so that items that do possess the feature will not be able to have a large value and the feature may disappear?
Empirical tests may be needed here. 

The approach is similar to Khan et al. 2014. "Scalable Collaborative Bayesian Preference Learning" but differs in that
we also place priors over the weights and model correlations between different items and between different people.
Our use of priors that encourage sparseness in the features is also different. 

Observed features -- why is it good to use them as inputs to latent features? 
-- we assume some patterns in the observations are common to multiple people, and these manifest as latent features
-- we can use the GP model to map observations to latent features to handle sparsity of data for each item
and person
-- the GP will model dependencies between the input features
An alternative would be a flat model, where the input features for items were added to columns of w, 
and the input features of people created new rows in y. This may make it easier to learn which features are relevant,
but does not help with sparse features because we could not use a GP to smooth and interpolate between items, so 
would need mode observed preference pairs for each item and person to determine their latent feature values.  

For testing effects of no. inducing points, forgetting rate, update size, delay, it would be useful to see accuracy and 
convergence rate.

Created on 2 Jun 2016

@author: simpson
"""

import numpy as np
# from sklearn.decomposition import FactorAnalysis
from scipy.stats import multivariate_normal as mvn, norm
import logging
from gp_classifier_vb import matern_3_2_from_raw_vals, derivfactor_matern_3_2_from_raw_vals, temper_extreme_probs, \
    check_convergence, diagonal_from_raw_vals, compute_median_lengthscales, derivfactor_diag_from_raw_vals
from gp_pref_learning import GPPrefLearning, get_unique_locations, pref_likelihood
from scipy.linalg import block_diag
from scipy.special import gammaln, psi
from scipy.stats import gamma
from scipy.optimize import minimize
# from joblib import Parallel, delayed
# import multiprocessing


def expec_output_scale(shape_s0, rate_s0, N, invK, f_mean, m, invK_f_cov=None, f_cov=None):
    # learn the output scale with VB
    shape_s = shape_s0 + 0.5 * N

    if invK_f_cov is None:
        if f_cov is None:
            logging.error('Provide either invK_f_cov or f_cov')
            return

        if invK.ndim == 1 and f_cov.ndim == 1:
            invK_f_cov = invK * f_cov
        else:
            invK_f_cov = invK.dot(f_cov)

    if invK.ndim == 1:
        if not np.isscalar(m):
            m = m.flatten()
        if not np.isscalar(f_mean):
            f_mean = f_mean.flatten()

        rate_s = rate_s0 + 0.5 * np.sum(invK_f_cov + invK * (f_mean - m)**2 )
    else:
        invK_expecFF = invK_f_cov + invK.dot((f_mean - m).dot(f_mean.T - m.T))
        rate_s = rate_s0 + 0.5 * np.trace(invK_expecFF)

    return shape_s, rate_s


def lnp_output_scale(shape_s0, rate_s0, shape_s, rate_s, s=None, Elns=None):
    if s is None:
        s = shape_s / rate_s
    if Elns is None:
        Elns = psi(shape_s) - np.log(rate_s)

    logprob_s = - gammaln(shape_s0) + shape_s0 * np.log(rate_s0) + (shape_s0 - 1) * Elns - rate_s0 * s
    return logprob_s


def lnq_output_scale(shape_s, rate_s, s=None, Elns=None):
    if s is None:
        s = shape_s / rate_s
    if Elns is None:
        Elns = psi(shape_s) - np.log(rate_s)

    lnq_s = - gammaln(shape_s) + shape_s * np.log(rate_s) + (shape_s - 1) * Elns - rate_s * s
    return lnq_s


def update_gaussian(invK, s, Sigma, x):
    cov = np.linalg.inv((invK * s) + Sigma)
    m = cov.dot(x)
    return m, cov

def expec_pdf_gaussian(K, invK, Elns, N, s, f, mu, f_cov, mu_cov):
    """
    Expected value of the PDF with respect to the function values with expectation f, the mean values with expectation
    mu, and the inverse covariance scale with expectation s.
    
    Parameters
    ----------
    
    :param K : covariance matrix (without scaling)
    :param invK : inverse of the covariance matrix (without scaling)
    :param Elns : expected log of the covariance inverse scale factor
    :param N : number of data points
    :param s : expected covariance inverse scale factor
    :param f : expected function values
    :param mu : expected mean values
    :param f_cov : covariance of the function values
    :param mu_cov : covariance of the mean parameters; this is needed if the mean is a model parameter inferred using VB
    """
    if K.ndim == 1:
        logdet_K = np.sum(np.log(K))
        invK_expecF = (s * invK) * np.diag(f_cov + (f - mu).dot((f - mu).T) + mu_cov)
        tr_invK_expecF = np.sum(invK_expecF)
    else:
        _, logdet_K = np.linalg.slogdet(K)
        invK_expecF = (s * invK).dot(f_cov + (f - mu).dot((f - mu).T) + mu_cov)
        tr_invK_expecF = np.trace(invK_expecF)
    logdet_Ks = - np.sum(N * Elns) + logdet_K
    logpf = 0.5 * (- np.log(2 * np.pi) * N - logdet_Ks - tr_invK_expecF)

    return logpf


def expec_q_gaussian(f_cov, D):
    """

    :param f_cov:
    :param D:
    :return:
    """
    if f_cov.ndim == 1:
        logdet_C = np.sum(np.log(f_cov))
    else:
        _, logdet_C = np.linalg.slogdet(f_cov)

    logqf = 0.5 * (- np.log(2 * np.pi) * D - logdet_C - D)
    return logqf


class CollabPrefLearningVB(object):
    """
    Model for analysing the latent personality features that affect each person's preferences. Inference using 
    variational Bayes.
    """

    def __init__(self, nitem_features, nperson_features=0, shape_s0=1, rate_s0=1, shape_st0=None, rate_st0=None,
                 shape_sy0=None, rate_sy0=None, shape_ls=1, rate_ls=100, ls=100, shape_lsy=1, rate_lsy=100, lsy=100,
                 verbose=False, nfactors=20, use_common_mean_t=True, kernel_func='matern_3_2', use_lb=True):
        """

        :param nitem_features:
        :param nperson_features:
        :param shape_s0:
        :param rate_s0:
        :param shape_ls:
        :param rate_ls:
        :param ls:
        :param shape_lsy:
        :param rate_lsy:
        :param lsy:
        :param verbose:
        :param nfactors:
        :param use_common_mean_t:
        :param kernel_func:
        """
        self.nitem_features = nitem_features
        self.nperson_features = nperson_features
        self.Nfactors = nfactors

        shape_s0 = float(shape_s0)
        rate_s0 = float(rate_s0)

        # too much variance leads to extreme values of E[s], which then leads to bad covariance matrices, then crashes
        self.shape_sw0 = shape_s0
        self.rate_sw0 = rate_s0 #* self.Nfactors

        self.shape_sy0 = shape_sy0 if shape_sy0 is not None else shape_s0
        self.rate_sy0 = rate_sy0 if rate_sy0 is not None else rate_s0#/ self.Nfactors

        self.shape_st0 = shape_st0 if shape_st0 is not None else shape_s0
        self.rate_st0 = rate_st0 if rate_st0 is not None else rate_s0 #* self.Nfactors # this rebalances the prior expected variance of t against the latent components

        # posterior moments
        self.shape_st = self.shape_st0
        self.rate_st = self.rate_st0

        # y has different length-scales because it is over user features space
        self.shape_ls = shape_ls
        self.rate_ls = rate_ls

        if ls is not None:
            self.n_wlengthscales = len(
                np.array([ls]).flatten())  # can pass in a single length scale to be used for all dimensions
        else:
            self.n_wlengthscales = self.nitem_features
        self.ls = ls

        self.shape_lsy = shape_lsy
        self.rate_lsy = rate_lsy
        self.lsy = lsy
        if lsy is not None:
            self.n_ylengthscales = len(
                np.array([lsy]).flatten())  # can pass in a single length scale to be used for all dimensions
        else:
            self.n_ylengthscales = self.nperson_features

        self.t_mu0 = 0

        self.uselowerbound = use_lb
        if use_lb:
            self.conv_threshold = 1e-4
        else:
            self.conv_threshold = 1e-2

        self.conv_check_freq = 2

        self.max_iter_G = 10
        self.max_iter = 200
        self.min_iter = 1
        self.n_converged = 10  # number of iterations while apparently converged (avoids numerical errors)
        self.vb_iter = 0

        self.verbose = verbose

        self.use_t = use_common_mean_t

        self._select_covariance_function(kernel_func)

        self.new_obs = True  # flag to indicate that the model has not yet been fitted

    def _select_covariance_function(self, cov_type):
        self.cov_type = cov_type
        if cov_type == 'matern_3_2':
            self.kernel_func = matern_3_2_from_raw_vals
            self.kernel_der = derivfactor_matern_3_2_from_raw_vals
        # the other kernels no longer work because they need to use kernel functions that work with the raw values
        elif cov_type == 'diagonal':
            self.kernel_func = diagonal_from_raw_vals
            self.kernel_der = derivfactor_diag_from_raw_vals
        else:
            logging.error('PreferenceComponents: Invalid covariance type %s' % cov_type)

    # FITTING --------------------------------------------------------------------------------------------------------

    def _scaled_Kw(self, K, s, s_cov, inv_scale):

        N = K.shape[0]
        Npeople = s.shape[1]

        scaledK = np.zeros((N * Npeople, N * Npeople))

        for f in range(self.Nfactors):
            fidxs = np.arange(Npeople) + f * Npeople
            scaling = s[f:f+1, :].T.dot(s[f:f+1, :]) + s_cov[fidxs, :][:, fidxs]

            scaling = scaling[None, :, :, None]
            scaledK_f = K[:, None, None, :] * scaling
            scaledK_f = scaledK_f.reshape(N, Npeople, N * Npeople)
            scaledK_f = np.swapaxes(scaledK_f, 0, 2)
            scaledK_f = scaledK_f.reshape(N * Npeople, N * Npeople)

            scaledK_f /= inv_scale[f]

            scaledK += scaledK_f

        return scaledK

    def _init_covariance(self):
        self.K = self.kernel_func(self.obs_coords, self.ls) # + np.eye(self.N) * 1e-6
        self.invK = np.linalg.inv(self.K)

        blocks = [self.K for _ in range(self.Nfactors)]
        self.Kw = block_diag(*blocks)
        self.invKw = np.linalg.inv(self.Kw)

        blocks = np.tile(self.K[None, :, :], (self.Npeople, 1, 1))
        self.Kt = block_diag(*blocks)

        if self.person_features is None:
            self.invKy_block = np.diag(np.ones(self.Npeople))  # they are all ones
            self.invKy = np.diag(np.ones(self.Npeople * self.Nfactors))  # they are all ones

            self.Ky_block = np.diag(np.ones(self.Npeople))  # they are all ones
            self.Ky = np.diag(np.ones(self.Npeople * self.Nfactors))  # they are all ones

        else:
            self.lsy = np.zeros(self.nperson_features) + self.lsy
            self.Ky_block = self.y_kernel_func(self.person_features, self.lsy) # + np.eye(self.Npeople) * 1e-6
            self.invKy_block = np.linalg.inv(self.Ky_block)

            blocks = [self.Ky_block for _ in range(self.Nfactors)]
            self.Ky = block_diag(*blocks)
            self.invKy = np.linalg.inv(self.Ky)

        self.shape_sw = np.zeros(self.Nfactors) + self.shape_sw0
        self.rate_sw = np.zeros(self.Nfactors) + self.rate_sw0
        self.shape_sy = np.zeros(self.Nfactors) + self.shape_sy0
        self.rate_sy = np.zeros(self.Nfactors) + self.rate_sy0

    def _init_w(self):
        self.sw_matrix = np.ones(self.Kw.shape) * self.shape_sw0 / self.rate_sw0

        # initialise the factors randomly -- otherwise they can get stuck because there is nothing to differentiate them
        # i.e. the cluster identifiability problem
        self.w = mvn.rvs(np.zeros(self.N), self.K, self.Nfactors).reshape(self.Nfactors, self.N).T #np.zeros((self.N, self.Nfactors))
        self.w /= (self.shape_sw/self.rate_sw)[None, :]
        self.wy = np.zeros((self.N * self.Npeople, 1))
        self.w_cov = self.Kw / self.sw_matrix

        self.Sigma_w = np.zeros((self.N, self.N, self.Nfactors))

        # self.sprior = self.Nfactors * self.shape_sw0 / self.rate_sw0 * self.shape_sy0 / self.rate_sy0
        self.sprior = self.shape_sw0 / self.rate_sw0

        if self.new_obs:
            self.wy_gp = GPPrefLearning(self.nitem_features, 0, self.shape_sw0, 1.0 / (self.sprior / self.shape_sw0), self.shape_ls, self.rate_ls, self.ls,
                                    fixed_s=True, kernel_func='pre', use_svi=False)
            # delay=self.delay, forgetting_rate=self.forgetting_rate,
            self.wy_gp.max_iter_VB_per_fit = 1
            self.wy_gp.min_iter_VB = 1
            self.wy_gp.max_iter_G = self.max_iter_G # G needs to converge within each VB iteration otherwise q(w) is very poor and crashes
            self.wy_gp.verbose = self.verbose
            self.wy_gp.conv_threshold = 1e-3
            self.wy_gp.conv_threshold_G = 1e-3
            self.wy_gp.conv_check_freq = 1

        # intialise Q using the prior covariance
        Kw = self._scaled_Kw(self.K, np.zeros((self.Nfactors, self.Npeople)), self.Ky / self.shape_sy0 * self.rate_sy0,
                             self.shape_sw / self.rate_sw)

        self.wy_gp.set_training_data(self.pref_v, self.pref_u, self.dummy_obs_coords, self.preferences,
                                     mu0=np.zeros((self.N*self.Npeople, 1)), K=Kw, input_type=self.input_type)

    def _init_t(self):
        self.t = np.zeros((self.N, 1))
        self.st = self.shape_st0 / self.rate_st0

        self.t_mu0 = np.zeros((self.N, 1)) + self.t_mu0

        if not self.use_t:
            return

        if self.new_obs:
            self.t_gp = GPPrefLearning(self.nitem_features, 0, 1, 1, self.shape_ls, self.rate_ls, self.ls,
                                       fixed_s=True, kernel_func='pre', use_svi=False)
            # delay=self.delay, forgetting_rate=self.forgetting_rate,
            self.t_gp.max_iter_VB_per_fit = 1
            self.t_gp.min_iter_VB = 1
            self.t_gp.max_iter_G = self.max_iter_G # G needs to converge within each VB iteration otherwise q(w) is very poor and crashes
            self.t_gp.verbose = self.verbose
            self.t_gp.conv_threshold = 1e-3
            self.t_gp.conv_threshold_G = 1e-3
            self.t_gp.conv_check_freq = 1

        self.t_gp.vb_iter = 0

    def _scaled_Ky(self, K, w, w_cov, inv_scale):

        N = w.shape[0]
        Npeople = K.shape[0]

        # Ky uses same layout as Kw
        scaledK = np.zeros((N * Npeople, N * Npeople))

        for f in range(self.Nfactors):
            fidxs = np.arange(N) + f * N
            #wscaling = np.diag(self.w[:, f]**2)# + np.diag(self.w_cov[fidxs, :][:, fidxs]))#
            scaling = w[:, f:f+1].dot(w[:, f:f+1].T) + w_cov[fidxs, :][:, fidxs]
            scaling = scaling[:, None, None, :]
            scaledK_f = K[None, :, :, None] * scaling

            scaledK_f = scaledK_f.reshape(N, Npeople, N * Npeople)
            scaledK_f = np.swapaxes(scaledK_f, 0, 2)
            scaledK_f = scaledK_f.reshape(N * Npeople, N * Npeople)

            scaledK_f /= inv_scale[f]

            scaledK += scaledK_f
        #scaledK += np.eye(scaledK.shape[0]) * 1e-6

        return scaledK

    def _init_y(self):
        self.y = mvn.rvs(np.zeros(self.Npeople), self.Ky_block, self.Nfactors).reshape(self.Nfactors, self.Npeople)
        self.y /= (self.shape_sy/self.rate_sy)[:, None]

        self.y_cov = self.Ky

        self.Sigma_y = np.zeros((self.Npeople, self.Npeople, self.Nfactors))

    def _init_params(self):
        if self.Nfactors is None or self.Npeople < self.Nfactors:  # not enough items or people
            self.Nfactors = self.Npeople

        self._init_covariance()

        # put all prefs into a single GP to get a good initial mean estimate t -- this only makes sense if we can also 
        # estimate w y in a sensibel way, e.g. through factor analysis?
        # self.pref_gp[person].fit(items_1_p, items_2_p, prefs_p, mu0_1=mu0_1, mu0_2=mu0_2, process_obs=self.new_obs)

        self.ls = np.zeros(self.nitem_features) + self.ls

        self._init_w()
        self._init_y()
        self._init_t()

    def _process_observations(self, personIDs=None, items_1_coords=None, items_2_coords=None, item_features=None,
            preferences=None, person_features=None, input_type='binary'):
        """
        Save the input data into the objects used for fitting.
        :return:
        """

        pref_values_in_input = np.unique(preferences)
        if input_type == 'binary' and np.sum((pref_values_in_input < 0) | (pref_values_in_input > 1)):
            raise ValueError(
                'Binary input preferences specified but the data contained the values %s' % pref_values_in_input)
        elif input_type == 'zero-centered' and np.sum(
                (pref_values_in_input < -1) | (pref_values_in_input > 1)):
            raise ValueError(
                'Zero-centered input preferences specified but the data contained the values %s' % pref_values_in_input)
        elif input_type == 'zero-centered':
            # convert them to [0,1]
            preferences += 1
            preferences /= 2.0
        elif input_type != 'binary':
            raise ValueError('input_type for preference labels must be either "binary" or "zero-centered"')

        self.preferences = np.array(preferences, copy=False)


        if personIDs is not None:  # process new data
            self.new_obs = True  # there are people we haven't seen before
            # deal only with the original IDs to simplify prediction steps and avoid conversions

            if item_features is None:
                self.obs_coords, pref_v, pref_u = get_unique_locations(items_1_coords, items_2_coords)
            else:
                self.obs_coords = np.array(item_features, copy=False)
                pref_v = np.array(items_1_coords, copy=False)
                pref_u = np.array(items_2_coords, copy=False)
            self.N = self.obs_coords.shape[0]

            self.personIDs = np.array(personIDs)
            if person_features is not None:
                self.person_features = np.array(person_features,
                                                copy=False)  # rows per person, columns for feature values
                self.nperson_features = self.person_features.shape[1]
                self.Npeople = self.person_features.shape[0]

                self.y_kernel_func = self.kernel_func
            else:
                self.nperson_features = 0
                upeople = np.unique(personIDs)
                self.Npeople = np.max(upeople).astype(int) + 1
                self.person_features = None
                self.y_kernel_func = None

            # IDs must be for unique item-user pairs
            self.pref_v = pref_v + (self.N * self.personIDs)
            self.pref_u = pref_u + (self.N * self.personIDs)

            self.tpref_v = pref_v
            self.tpref_u = pref_u

            # the covariance matrices are pre-computed so obs_coords is not needed
            self.dummy_obs_coords = np.empty((self.N * self.Npeople, 1))
            self.tdummy_obs_coords = np.empty((self.N, 1))
        else:
            self.new_obs = False  # do we have new data? If so, reset everything. If not, don't reset the child GPs.

        self.input_type = input_type

        self.z = preferences.flatten()[:, np.newaxis].astype(float)
        self.nobs = self.z.shape[0]

    def fit(self, personIDs=None, items_1_coords=None, items_2_coords=None, item_features=None,
            preferences=None, person_features=None, optimize=False, maxfun=20, use_MAP=False, nrestarts=1,
            input_type='binary', use_median_ls=False):
        """
        Learn the model with data using variational Bayes.

        :param personIDs: a list of the person IDs of the people who expressed their preferences. These should ideally
        start from zero and be contiguous values, since they are used as indexes into matrices. Gaps in the IDs will
        mean wasted memory.
        :param items_1_coords: if item_features is None, these should be coordinates of the first items in the pairs
        being compared, otherwise these should be indexes into the item_features vector
        :param items_2_coords: if item_features is None, these should be coordinates of the second items in each pair
        being compared, otherwise these should be indexes into the item_features vector
        :param item_features: feature values for the items. Can be None if the items_x_coords provide the feature values as
        coordinates directly.
        :param preferences: the values, 0 or 1 to express that item 1 was preferred to item 2.
        :param person_features:
        :param optimize:
        :param maxfun:
        :param use_MAP:
        :param nrestarts:
        :param input_type:
        :param use_lb:
        :return:
        """
        if optimize:
            return self._optimize(personIDs, items_1_coords, items_2_coords, item_features, preferences,
                                  person_features, maxfun, use_MAP, nrestarts, input_type, use_median_ls)

        # if personIDs is not none, we assume this is new data being passed in
        self._process_observations(personIDs, items_1_coords, items_2_coords, item_features, preferences,
                                   person_features, input_type)

        if use_median_ls and personIDs is not None:
            self.ls = compute_median_lengthscales(self.obs_coords)
            if self.person_features is not None:
                self.lsy = compute_median_lengthscales(self.person_features)

        self._init_params()

        # reset the iteration counters
        self.vb_iter = 0
        old_w = np.inf
        old_y = np.inf
        old_lb = -np.inf
        converged_count = 0
        while ((self.vb_iter < self.min_iter) or (converged_count < self.n_converged)) and (
                self.vb_iter < self.max_iter):

            # set the value of t
            self._expec_t()

            # find the latent components
            self._expec_w(update_G=False if self.use_t else True)
            self._expec_y(update_G=False)

            self.new_obs = False # observations have now been processed once, only updates are required

            if not np.mod(self.vb_iter, self.conv_check_freq) == 0:
                converged = False # don't check anything
            elif self.uselowerbound :
                lb = self.lowerbound()
                converged = check_convergence(lb, old_lb, self.conv_threshold, True,
                                              self.vb_iter, self.verbose,
                                                   'CollabPL VB lower bound')

                old_lb = lb
            else:
                converged_w = check_convergence(self.w, old_w, self.conv_threshold, False,
                                                self.vb_iter, self.verbose,
                                                     'CollabPL VB, w ', change_as_a_fraction=False)
                converged_y = check_convergence(self.y, old_y, self.conv_threshold, False,
                                                self.vb_iter, self.verbose,
                                                     'CollabPL VB, y ', change_as_a_fraction=False)
                converged = converged_w & converged_y

                old_w = self.w.copy()
                old_y = self.y.copy()

            if converged:
                converged_count += 1
            elif converged_count > 0 and np.mod(self.vb_iter, self.conv_check_freq) == 0:  # reset the convergence count as the difference has increased again
                converged_count = 0

            self.vb_iter += 1

            if self.verbose:
                logging.debug('Converged count = %i' % converged_count)

        logging.debug("Converged in %i iterations. Running final update..." % self.vb_iter)

        # Do a final iteration with the latest values of s so that predictions and inducing point distributions all used
        # matching values of s for their kernels.
        self._expec_t(update_s=False, update_G=False)
        self._expec_w(update_s=False, update_G=False)
        self._expec_y(update_s=False, update_G=False)

        if self.verbose:
            logging.debug('Completed final update.')

    def _expec_t(self, update_s=True, update_G=True):

        if not self.use_t:
            return

        mu0 = self.w.dot(self.y).T.reshape(self.N * self.Npeople, 1)
        #mu0 = np.zeros((self.N * self.Npeople, 1))

        self.t_gp.s = self.st
        self.t_gp.fit(self.pref_v, self.pref_u, self.dummy_obs_coords, self.preferences,
                      mu0=mu0, K=self.Kt,
                      process_obs=self.new_obs, input_type=self.input_type)

        invQ = self.t_gp.get_obs_precision()
        invQ = invQ.reshape(self.Npeople * self.N, self.Npeople, self.N)
        invQ = np.swapaxes(invQ, 0, 2).reshape(self.N, self.Npeople, self.Npeople, self.N)
        t_prec = np.sum(np.sum(invQ, 2), 1)  # Npeople x Npeople

        z0 = self.t_gp.forward_model(self.t_gp.obs_f) + self.t_gp.G.dot(self.t_gp.mu0 - self.t_gp.obs_f)
        invQ_f = self.t_gp.G.T.dot(np.diag(1.0 / self.t_gp.Q)).dot(self.t_gp.z - z0)  # this subtracts out the t prior
        x = np.sum(invQ_f.reshape(self.Npeople, self.N), 0)
        x = x.reshape(self.N, 1) + self.t_mu0

        self.t, self.t_cov = update_gaussian(self.invK, self.st, t_prec, x)


        if update_s:
            self.shape_st, self.rate_st = expec_output_scale(self.shape_st0, self.rate_st0, self.N,
                                                         self.invK, self.t, np.zeros((self.N, 1)),
                                                         f_cov=self.t_cov)
            self.st = self.shape_st / self.rate_st

    def _expec_w(self, update_s=True, update_G=-1):
        """
        Compute the expectation over the latent features of the items and the latent personality components
        :return:
        """

        Kw = self._scaled_Kw(self.K, self.y, self.y_cov, self.shape_sw / self.rate_sw)

        t = np.tile(self.t, (self.Npeople, 1))

        self.wy_gp.fit(self.pref_v, self.pref_u, self.dummy_obs_coords, self.preferences, mu0=t, K=Kw,
                       process_obs=False, input_type=self.input_type)

        # compute sigma_w
        invQ = self.wy_gp.get_obs_precision()
        invQ = invQ.reshape(self.Npeople * self.N, self.Npeople, self.N)
        invQ = np.swapaxes(invQ, 0, 2).reshape(self.N, self.Npeople, self.Npeople, self.N)

        w_prec = np.zeros((self.N * self.Nfactors, self.N * self.Nfactors))
        for f in range(self.Nfactors):
            for g in range(self.Nfactors):
                #  is to update each factor in turn, which may do a better job of cancelling out unneeded factors.
                yscaling = self.y[f:f+1, :].T.dot(self.y[g:g+1, :]) + self.y_cov[f * self.Npeople + np.arange(self.Npeople), :]\
                                                                  [:, g * self.Npeople + np.arange(self.Npeople)]
                #yscaling = np.diag(self.y[f, :]**2 + np.diag(self.y_cov[f * self.Npeople + np.arange(self.Npeople), :]\
                #                                                  [:, g * self.Npeople + np.arange(self.Npeople)]))

                Sigma_f_g = np.sum(np.sum(yscaling[None, :, :, None] * invQ, 2), 1) # Npeople x Npeople

                fidxs = np.tile(f * self.N + np.arange(self.N)[:, None], (1, self.N))
                gidxs = np.tile(g * self.N + np.arange(self.N)[None, :], (self.N, 1))
                w_prec[fidxs, gidxs] = Sigma_f_g

                if f == g:
                    self.Sigma_w[:, :, f] = Sigma_f_g

        z0 = self.wy_gp.forward_model(self.wy_gp.obs_f) + self.wy_gp.G.dot(self.wy_gp.mu0 - self.wy_gp.obs_f)
        invQ_f = self.wy_gp.G.T.dot(np.diag(1.0 / self.wy_gp.Q)).dot(self.wy_gp.z - z0)  # this subtracts out the t prior
        x = self.y.dot(invQ_f.reshape(self.Npeople, self.N))
        x = x.reshape(self.N * self.Nfactors, 1)

        # The prediction of w using this method is half the magnitude of obs_f when we have 1 person, 1 factor, y=1
        # The values of the covariance are also half the size.
        # This is correct because the ratio of (y*y + var(y)) / y = 2? The prior covariance for w_gp is twice as large as for the update below,
        # and invQ is similarly divided, mqking w_prec half the size of Q.  Why isn't the resulting self.w 2**2 times smaller
        # than self.w_gp.obs_f? Because of the y*y terms cancelling, that leads us to not needing var(y) in the lines above.
        self.w, self.w_cov = update_gaussian(self.invKw, self.sw_matrix, w_prec, x)
        self.w = np.reshape(self.w, (self.Nfactors, self.N)).T  # w is N x Nfactors

        if not update_s:
            return

        for f in range(self.Nfactors):
            fidxs = np.arange(self.N) + (self.N * f)
            self.shape_sw[f], self.rate_sw[f] = expec_output_scale(self.shape_sw0, self.rate_sw0, self.N,
                                                                   self.invK, self.w[:, f:f + 1], np.zeros((self.N, 1)),
                                                                   f_cov=self.w_cov[fidxs, :][:, fidxs])

            self.sw_matrix[fidxs, :] = self.shape_sw[f] / self.rate_sw[f]

    def _expec_y(self, update_s=True, update_G=-1):
        """
        Compute expectation over the personality components using VB

        # Still a problem when expec_w is switched off!
        # When y and w are both fixed, the likelihod still goes down -- why?

        :return:
        """

        Ky = self._scaled_Ky(self.Ky_block, self.w, self.w_cov, self.shape_sy / self.rate_sy)

        t = np.tile(self.t, (self.Npeople, 1))

        self.wy_gp.fit(self.pref_v, self.pref_u, self.dummy_obs_coords, self.preferences, mu0=t, K=Ky,
                       process_obs=False, input_type=self.input_type)

        # This is to compute q(y)
        invQ = self.wy_gp.get_obs_precision()
        invQ = invQ.reshape(self.Npeople * self.N, self.Npeople, self.N)
        invQ = np.swapaxes(invQ, 0, 2).reshape(self.N, self.Npeople, self.Npeople, self.N)

        y_prec = np.zeros((self.Nfactors * self.Npeople, self.Nfactors * self.Npeople))
        for f in range(self.Nfactors):
            w_cov_f = self.w_cov[f * self.N + np.arange(self.N), :]
            for g in range(self.Nfactors):
                wscaling = w_cov_f[:, g * self.N + np.arange(self.N)] + self.w[:, f:f+1].dot(self.w[:, g:g+1].T)

                Sigma_f_g = np.sum(np.sum(wscaling[:, None, None, :] * invQ, 3), 0) # Npeople x Npeople

                fidxs = np.tile(f * self.Npeople + np.arange(self.Npeople)[:, None], (1, self.Npeople))
                gidxs = np.tile(g * self.Npeople + np.arange(self.Npeople)[None, :], (self.Npeople, 1))

                y_prec[fidxs, gidxs] = Sigma_f_g

                if f == g:
                    self.Sigma_y[:, :, f] = Sigma_f_g

        z0 = self.wy_gp.forward_model(self.wy_gp.obs_f) + self.wy_gp.G.dot(self.wy_gp.mu0 - self.wy_gp.obs_f)
        invQ_f = self.wy_gp.G.T.dot(np.diag(1.0 / self.wy_gp.Q)).dot(self.wy_gp.z - z0)  # this subtracts out the t prior
        x = self.w.T.dot(invQ_f.reshape(self.Npeople, self.N).T) # here we sum over items
        x = x.reshape(self.Npeople * self.Nfactors, 1)

        self.y, self.y_cov = update_gaussian(self.invKy, 1, y_prec, x)
        self.y = np.reshape(self.y, (self.Nfactors, self.Npeople))  # y is Npeople x Nfactors

        # for f in range(self.Nfactors):
        #     fidxs = np.arange(self.Npeople) + (self.Npeople * f)
        #     self.shape_sy[f], self.rate_sy[f] = expec_output_scale(self.shape_sy0, self.rate_sy0, self.Npeople,
        #                                                            self.invKy_block, self.y[f:f + 1, :].T,
        #                                                            np.zeros((self.Npeople, 1)),
        #                                                            f_cov=self.y_cov[fidxs, :][:, fidxs])

    def _logpD(self):
        fmean = (self.w.dot(self.y) + self.t).T.reshape(self.N * self.Npeople, 1)
        rho = pref_likelihood(fmean, v=self.pref_v, u=self.pref_u)
        rho = temper_extreme_probs(rho)
        logrho = np.log(rho)
        lognotrho = np.log(1 - rho)

        # prod_cov = 0
        # y_w_cov_y = 0
        # w_y_cov_w = 0
        # for f in range(self.Nfactors):
        #
        #     fidxs = np.arange(self.N) + (self.N * f)
        #     w_cov = self.w_cov[fidxs, :][:, fidxs]
        #
        #     fidxs = np.arange(self.Npeople) + (self.Npeople * f)
        #     y_cov = self.y_cov[fidxs, :][:, fidxs]
        #
        #     cov = w_cov[None, :, :, None] * y_cov[:, None, None, :]
        #     cov = cov.reshape(self.N * self.Npeople, self.N * self.Npeople)
        #
        #     y_w_cov_y_f = w_cov[None, :, :, None] * self.y[f:f+1, :].T.dot(self.y[f:f+1, :])[:, None, None, :]
        #     y_w_cov_y_f = y_w_cov_y_f.reshape(self.N * self.Npeople, self.N * self.Npeople)
        #     y_w_cov_y += y_w_cov_y_f
        #
        #     w_y_cov_w_f = y_cov[:, None, None, :] * self.w[:, f:f+1].dot(self.w[:, f:f+1].T)[None, :, :, None]
        #     w_y_cov_w_f = w_y_cov_w_f.reshape(self.N * self.Npeople, self.N * self.Npeople)
        #     w_y_cov_w += w_y_cov_w_f
        #
        #     prod_cov += cov

        # we want to replace this precision with the global values. w_gp and y_gp are different
        # because G is different in each.
        data_ll = self.wy_gp.data_ll(logrho, lognotrho)
        #data_ll -= 0.5 * np.trace((prod_cov + w_y_cov_w + y_w_cov_y).dot(self.wy_gp.get_obs_precision()))

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
        logpw = expec_pdf_gaussian(self.Kw, self.invKw, Elnsw, self.N, self.sw_matrix,
                       self.w.T.reshape(self.N * self.Nfactors, 1), 0, self.w_cov, 0)
        # f_cov=self.w_cov seems not needed because it simplifies with a term in the likelihood to D -- but this doesn't
        # work here until w and y have converged because they are not yet consistent, hence there are differences in the
        # terms that should cancel.

        logqw = expec_q_gaussian(self.w_cov, self.N * self.Nfactors)

        if self.use_t:
            logpt = expec_pdf_gaussian(self.K, self.invK, Elnst, self.N, st, self.t, self.t_mu0,
                                       0, 0) - 0.5 * self.N # for t, the trace terms should simplify with those in the
            # the likelihood. This works because they do not depend on scaling terms that are also being learned in the
            # same loop.
            logqt = expec_q_gaussian(self.t_cov, self.N)
        else:
            logpt = 0
            logqt = 0

        logpy = expec_pdf_gaussian(self.Ky, self.invKy, Elnsy, self.Npeople, 1,
                   self.y.reshape(self.Npeople * self.Nfactors, 1), 0, self.y_cov, 0)
        logqy = expec_q_gaussian(self.y_cov, self.Npeople * self.Nfactors)

        logps_y = 0
        logqs_y = 0
        logps_w = 0
        logqs_w = 0
        for f in range(self.Nfactors):
            logps_w += lnp_output_scale(self.shape_sw0, self.rate_sw0, self.shape_sw[f], self.rate_sw[f], sw[f],
                                        Elnsw[f])
            logqs_w += lnq_output_scale(self.shape_sw[f], self.rate_sw[f], sw[f], Elnsw[f])

            logps_y += lnp_output_scale(self.shape_sy0, self.rate_sy0, self.shape_sy[f], self.rate_sy[f], sy[f],
                                        Elnsy[f])
            logqs_y += lnq_output_scale(self.shape_sy[f], self.rate_sy[f], sy[f], Elnsy[f])

        logps_t = lnp_output_scale(self.shape_st0, self.rate_st0, self.shape_st, self.rate_st, st, Elnst)
        logqs_t = lnq_output_scale(self.shape_st, self.rate_st, st, Elnst)

        w_terms = logpw - logqw + logps_w - logqs_w
        y_terms = logpy - logqy + logps_y - logqs_y
        t_terms = logpt - logqt + logps_t - logqs_t

        lb = data_ll + t_terms + w_terms + y_terms

        if self.verbose:
            logging.debug('s_w=%s' % (self.shape_sw / self.rate_sw))
            logging.debug('s_y=%s' % (self.shape_sy / self.rate_sy))
            logging.debug('s_t=%.2f' % (self.shape_st / self.rate_st))

        if self.verbose:
            logging.debug('likelihood=%.3f, wterms=%.3f, yterms=%.3f, tterms=%.3f' % (data_ll, w_terms, y_terms, t_terms))
            logging.debug("Iteration %i: Lower bound = %.3f, " % (self.vb_iter, lb))
            logging.debug("t: %.2f, %.2f" % (np.min(self.t), np.max(self.t)))
            logging.debug("w: %.2f, %.2f" % (np.min(self.w), np.max(self.w)))
            logging.debug("y: %.2f, %.2f" % (np.min(self.y), np.max(self.y)))

        return lb

    def lowerbound_sampling(self):
        """
        An alternative method for computing the lower bound using sampling. However, it seems to give poor results,
        so there may be an error in here due to the sampling complexity, or due to some of the samples producing
        probabilities close to 0 or 1 that become extreme log values, which distort the DLL term.
        :return:
        """

        # Uncomment entire block to use sampling instead of the analytical estimates of rho -- useful to debug predict
        nsamples = 10000

        w_samples = mvn.rvs(mean=self.w.T.reshape(self.N * self.Nfactors),
                                     cov=self.w_cov, size=(nsamples)).T

        y_samples = mvn.rvs(mean=self.y.reshape(self.Npeople * self.Nfactors),
                                     cov=self.y_cov, size=(nsamples)).T

        if self.use_t:
            t_samples = np.random.normal(loc=self.t,
                                     scale=np.sqrt(np.diag(self.t_cov))[:, None],
                                     size=(self.N, nsamples))
        else:
            t_samples = np.zeros((self.N, nsamples))

        w_samples = w_samples.reshape(self.Nfactors, self.N, nsamples)
        y_samples = y_samples.reshape(self.Nfactors, self.Npeople, nsamples)

        f_samples = [(w_samples[:, :, i].T.dot(y_samples[:, :, i]) + t_samples[:, None, i])
                         .T.reshape(self.N * self.Npeople) for i in range(nsamples)]
        f_samples = np.array(f_samples).T

        g_f = (f_samples[self.pref_v, :] - f_samples[self.pref_u, :]) / np.sqrt(2)
        phi = norm.cdf(g_f)  # the probability of the actual observation, which takes g_f as a parameter. In the

        # rho = np.mean(phi, axis=1)[:, np.newaxis]

        phi = temper_extreme_probs(phi)

        logphi = np.log(phi)
        lognotphi = np.log(1 - phi)

        logrho = np.mean(logphi, axis=1)[:, np.newaxis]
        lognotrho = np.mean(lognotphi, axis=1)[:, np.newaxis]

        data_ll = self.wy_gp.data_ll(logrho, lognotrho)

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
        logpw = expec_pdf_gaussian(self.Kw, self.invKw, Elnsw, self.N, self.sw_matrix,
                       self.w.T.reshape(self.N * self.Nfactors, 1), 0, self.w_cov, 0) #- 0.5 * self.N #* self.Npeople
        # f_cov=self.w_cov not needed because it simplifies with a term in the likelihood to D

        logqw = expec_q_gaussian(self.w_cov, self.N * self.Nfactors)

        if self.use_t:
            logpt = expec_pdf_gaussian(self.K, self.invK, Elnst, self.N, st, self.t, self.t_mu0,
                                       self.t_cov, 0) #- 0.5 * self.N #* self.Npeople
            # f_cov=self.t_cov not needed because it simplifies with a term in the likelihood to D
            logqt = expec_q_gaussian(self.t_cov, self.N)
        else:
            logpt = 0
            logqt = 0

        logpy = expec_pdf_gaussian(self.Ky, self.invKy, Elnsy, self.Npeople, 1,
                   self.y.reshape(self.Npeople * self.Nfactors, 1), 0, self.y_cov, 0) #- 0.5 * self.Npeople# * self.N
        # f_cov=self.y_cov not needed because it simplifies with a term in the likelihood to D
        logqy = expec_q_gaussian(self.y_cov, self.Npeople * self.Nfactors)

        logps_y = 0
        logqs_y = 0
        logps_w = 0
        logqs_w = 0
        for f in range(self.Nfactors):
            logps_w += lnp_output_scale(self.shape_sw0, self.rate_sw0, self.shape_sw[f], self.rate_sw[f], sw[f],
                                        Elnsw[f])
            logqs_w += lnq_output_scale(self.shape_sw[f], self.rate_sw[f], sw[f], Elnsw[f])

            logps_y += lnp_output_scale(self.shape_sy0, self.rate_sy0, self.shape_sy[f], self.rate_sy[f], sy[f],
                                        Elnsy[f])
            logqs_y += lnq_output_scale(self.shape_sy[f], self.rate_sy[f], sy[f], Elnsy[f])

        logps_t = lnp_output_scale(self.shape_st0, self.rate_st0, self.shape_st, self.rate_st, st, Elnst)
        logqs_t = lnq_output_scale(self.shape_st, self.rate_st, st, Elnst)

        w_terms = logpw - logqw + logps_w - logqs_w
        y_terms = logpy - logqy + logps_y - logqs_y
        t_terms = logpt - logqt + logps_t - logqs_t

        lb = data_ll + t_terms + w_terms + y_terms

        if self.verbose:
            logging.debug('s_w=%s' % (self.shape_sw / self.rate_sw))
            logging.debug('s_y=%s' % (self.shape_sy / self.rate_sy))
            logging.debug('s_t=%.2f' % (self.shape_st / self.rate_st))

        logging.debug('likelihood=%.3f, wterms=%.3f, yterms=%.3f, tterms=%.3f' % (data_ll, w_terms, y_terms, t_terms))

        if self.verbose:
            logging.debug("Iteration %i: Lower bound = %.3f, " % (self.vb_iter, lb))

        logging.debug("t: %.2f, %.2f" % (np.min(self.t), np.max(self.t)))
        logging.debug("w: %.2f, %.2f" % (np.min(self.w), np.max(self.w)))
        logging.debug("y: %.2f, %.2f" % (np.min(self.y), np.max(self.y)))

        return lb

    # PREDICTION --------------------------------------------------------------------------------------------------

    def predict_pairs_from_features(self, personids, item_0_feats=None, item_1_feats=None, person_feats=None, no_var=True):
        """
        Predict pairwise labels by passing in two lists of feature vectors for the first and second items in the pairs.
        This is different to the usual predict() method, which predicts pairwise labels given the indices of items into
        a set of features.

        """
        item_feats, item_0_idxs, item_1_idxs, mu0_out = get_unique_locations(item_0_feats, item_1_feats)
        return self.predict(personids, item_0_idxs, item_1_idxs, item_feats, person_feats, no_var)

    def predict(self, personids, item_0_idxs, item_1_idxs, item_features=None, person_features=None, no_var=False):

        if person_features is None:
            logging.info('Predict: no person features provided -- assuming the same people as seen in training')
            Npeople = len(np.unique(personids))
        else:
            Npeople = person_features.shape[0]
        personids = np.array(personids)

        if item_features is None:
            logging.info('Predict: no item features provided -- assuming same items as seen in training')
            N = self.N
        else:
            N = item_features.shape[0]

        # set personids and itemids to None to compute all pairs
        mu, f_cov, personids = self.predict_f(item_features, person_features, personids,
                                          return_cov=True, cov_0=item_0_idxs, cov_1=item_1_idxs, return_personids=True)

        mu = mu.T.reshape(N * Npeople, 1)

        pref_v = item_0_idxs + (N * personids)
        pref_u = item_1_idxs + (N * personids)

        # inserting the f_cov in here means we approximate the posterior over f=wy+t, which is product Gaussian,
        # with a Gaussian.

        if no_var:
            predicted_prefs = pref_likelihood(mu, subset_idxs=[], v=pref_v, u=pref_u)
        else:

            predicted_prefs = pref_likelihood(mu, f_cov[personids, item_0_idxs, item_0_idxs]
                                              + f_cov[personids, item_1_idxs, item_1_idxs]
                                              - f_cov[personids, item_0_idxs, item_1_idxs]
                                              - f_cov[personids, item_1_idxs, item_0_idxs],
                                              subset_idxs=[], v=pref_v, u=pref_u)
        predicted_prefs = temper_extreme_probs(predicted_prefs)

        return predicted_prefs

    def predict_f_item_person(self, itemids, personids, item_features=None, person_features=None):
        predicted_f, personids = self.predict_f(item_features, person_features, personids, return_personids=True)
        predicted_f = predicted_f[itemids, personids]
        return predicted_f

    def _y_var(self):
        return np.diag(self.y_cov)

    def _predict_y_tr(self, return_cov):
        y = self.y
        if return_cov:
            y_var = self._y_var()
        else:
            y_var = None

        return y, y_var

    def _compute_cov_w(self, _, __):
        cov_w = np.zeros((self.Nfactors, self.N, self.N))
        for f in range(self.Nfactors):
            fidxs = np.arange(self.N) + self.N * f
            cov_w[f] = self.w_cov[fidxs, :][:, fidxs]

        return cov_w

    def _compute_cov_t(self, _, __):
        return self.t_cov

    def _wy(self, w= None, y=None):

        if w is None:
            w = self.w
        if y is None:
            y = self.y

        return w.dot(y)

    def predict_f(self, item_features=None, person_features=None, personids=None, return_cov=False, cov_0=None,
                  cov_1=None, return_personids=False):

        if personids is not None:
            personids = np.array(personids).astype(int)

        if person_features is None:
            # Make predictions for all the users seen during training
            y, var_y = self._predict_y_tr(return_cov)

            if personids is not None:
                # get the subset of people specified by personids
                upeople = np.unique(personids)
                Npeople = len(upeople)

                Nfactors = self.Nfactors if not self.personal_component else self.Nfactors - self.Npeople

                # reuse the training points
                y_sub = np.zeros((Nfactors, Npeople))

                var_y_sub = np.ones((Nfactors, Npeople))

                # for any people IDs we have seen before, we use the posterior mean and variance
                people_from_training = np.in1d(upeople, self.personIDs)

                if np.any(people_from_training): # reuse any previously-seen people
                    # copy in the posterior means
                    y_sub[:, people_from_training] = y[:, upeople[people_from_training]]

                if return_cov:
                    for f in range(Nfactors):
                        var_y_sub[f, people_from_training] = var_y[f, upeople[people_from_training]]

                y = y_sub
                var_y = var_y_sub

                people_from_training = np.argwhere(people_from_training).flatten()
            else:
                people_from_training = np.arange(self.Npeople)
        else:
            # Make predictions for new users
            y, cov_y = self._predict_y(person_features, return_cov)
            if return_cov:
                var_y = cov_y[:, np.arange(y.shape[1]), np.arange(y.shape[1])]

            people_from_training = []

        if item_features is None:
            # reuse the training points
            t = self.t
            w = self.w

            if return_cov:
                cov_w = self._compute_cov_w(cov_0, cov_1)
                cov_t = self._compute_cov_t(cov_0, cov_1)
        else:
            t, w, cov_t, cov_w = self._predict_w_t(item_features, return_cov)

        N = w.shape[0]
        Npeople = y.shape[1]


        predicted_f = (self._wy(w, y, people_from_training) + t)

        # we may have reused some people from the training set, plus there may be some new user IDs.
        # The output from predict_f will contain only the people in personids.
        _, personids = np.unique(personids, return_inverse=True)

        if not return_cov:
            if not return_personids:
                return predicted_f
            else:
                return predicted_f, personids

        cov_f = np.zeros((Npeople, N, N))
        if self.use_t:
            cov_f += cov_t[None, :, :]

        # covariance of a product of two independent gaussians (product-Gaussian distribution)
        for f in range(self.Nfactors):
            cov_f_f = np.zeros((Npeople, N, N))

            if self.personal_component and f >= self.Nfactors - self.Npeople:
                cov_w_f = cov_w[f][None, :, :]
                cov_y_f = 0
            else:
                cov_f_f += var_y[f][:, None, None] * cov_w[f][None, :, :]

                yscaling = y[f, :]**2
                cov_wf = cov_w[f][None, :, :] * yscaling[:, None, None]

                wscaling = (w[:, f:f + 1].dot(w[:, f:f + 1].T))
                cov_yf = var_y[f][:, None, None] * wscaling[None, :, :]

            cov_f += cov_f_f + cov_wf + cov_yf

        # return predicted_f, t, cov_t, w, cov_w, y, cov_y
        if not return_personids:
            return predicted_f, cov_f
        else:
            return predicted_f, cov_f, personids

    def _predict_w_t(self, coords_1, return_cov=True):
        # kernel between pidxs and t
        K = self.kernel_func(coords_1, self.ls, self.obs_coords)
        K_starstar = self.kernel_func(coords_1, self.ls, coords_1)
        covpair = K.dot(self.invK)
        N = coords_1.shape[0]

        if self.use_t:
            # use kernel to compute t
            invKt = self.invK.dot(self.t)
            t_out = K.dot(invKt)

            if return_cov:
                cov_t = K_starstar * self.rate_st / self.shape_st + covpair.dot(self.t_cov - self.K * self.st).dot(covpair.T)
            else:
                cov_t = None
        else:
            t_out = np.zeros((N, 1))

            if return_cov:
                cov_t = np.zeros((N, N))
            else:
                cov_t = None

        w_out = K.dot(self.invK.dot(self.w))  # N x Nfactors

        if return_cov:
            cov_w = np.zeros((self.Nfactors, N, N))
            for f in range(self.Nfactors):
                fidxs = np.arange(self.N) + self.N * f
                cov_w[f] = K_starstar  * self.rate_sw[f] / self.shape_sw[f] + covpair.dot(self.w_cov[fidxs, :][:, fidxs]
                                                        - self.K * self.rate_sw[f] / self.shape_sw[f]).dot(covpair.T)
        else:
            cov_w = None

        return t_out, w_out, cov_t, cov_w

    def predict_common(self, item_features, item_0_idxs, item_1_idxs):
        '''
        Predict the common consensus values using t.
        '''
        if not self.use_t:
            return np.zeros(len(item_0_idxs))

        K = self.kernel_func(item_features, self.ls, self.obs_coords)
        K_starstar = self.kernel_func(item_features, self.ls, item_features)
        covpair = K.dot(self.invK)
        invKt = self.invK.dot(self.t)

        t_out = K.dot(invKt)
        cov_t = K_starstar * self.rate_st / self.shape_st + covpair.dot(self.t_cov + self.K * self.st).dot(covpair.T)

        predicted_prefs = pref_likelihood(t_out, cov_t[item_0_idxs, item_1_idxs]
                                          + cov_t[item_0_idxs, item_1_idxs]
                                          - cov_t[item_0_idxs, item_1_idxs]
                                          - cov_t[item_0_idxs, item_1_idxs],
                                          subset_idxs=[], v=item_0_idxs, u=item_1_idxs)

        return predicted_prefs

    def _predict_y(self, person_features, return_cov=True):

        Ky = self.y_kernel_func(person_features, self.lsy, self.person_features)
        Ky_starstar = self.y_kernel_func(person_features, self.lsy, person_features)

        covpair = Ky.dot(self.invKy_block)
        Npeople = person_features.shape[0]

        y_out = Ky.dot(self.invKy_block.dot(self.y.T)).T  # Nfactors x N

        if return_cov:
            cov_y = np.zeros((self.Nfactors, Npeople, Npeople))
            for f in range(self.Nfactors):
                fidxs = np.arange(self.Npeople) + self.Npeople * f
                cov_y[f] = Ky_starstar * self.rate_sy[f] / self.shape_sy[f] + covpair.dot(self.y_cov[fidxs, :][:, fidxs]
                                                    - self.Ky_block * self.rate_sy[f] / self.shape_sy[f]).dot(covpair.T)
        else:
            cov_y = None

        return y_out, cov_y

    # OPTIMIZATION ------------------------------------------------------------------------------------------------

    def _optimize(self, personIDs, items_1_coords, items_2_coords, item_features, preferences, person_features=None,
                  maxfun=20, use_MAP=False, nrestarts=1, input_type='binary', use_median_ls=False):

        max_iter = self.max_iter
        self.max_iter = 1
        self.fit(personIDs, items_1_coords, items_2_coords, item_features, preferences, person_features,
                 input_type=input_type, use_median_ls=use_median_ls)
        self.max_iter = max_iter

        min_nlml = np.inf
        best_opt_hyperparams = None
        best_iter = -1

        logging.debug("Optimising item length-scale for all dimensions")

        nfits = 0  # number of calls to fit function

        # optimise each length-scale sequentially in turn
        final_r = 0
        for r in range(nrestarts):
            # try to do it using the conjugate gradient method instead. Requires Jacobian (gradient) of LML 
            # approximation. If we also have Hessian or Hessian x arbitrary vector p, we can use Newton-CG, dogleg, 
            # or trust-ncg, which may be faster still?
            if person_features is None:
                initialguess = np.log(self.ls)
                logging.debug("Initial item length-scale guess in restart %i: %s" % (r, self.ls))
                res = minimize(self.neg_marginal_likelihood, initialguess, args=('item', -1, personIDs, items_1_coords,
                    items_2_coords, item_features, preferences, person_features, input_type, use_median_ls, use_MAP),
                    jac=self.nml_jacobian, method='L-BFGS-B', options={'maxiter': maxfun, 'ftol' : 1e-3,
                    'gtol': 10 ** (-self.nitem_features - 1)})
            else:
                initialguess = np.append(np.log(self.ls), np.log(self.lsy))
                logging.debug("Initial item length-scale guess in restart %i: %s" % (r, self.ls))
                logging.debug("Initial person length-scale guess in restart %i: %s" % (r, self.lsy))
                res = minimize(self.neg_marginal_likelihood, initialguess, args=('both', -1, personIDs, items_1_coords,
                    items_2_coords, item_features, preferences, person_features, input_type, use_median_ls, use_MAP,),
                    jac=self.nml_jacobian, method='L-BFGS-B', options={'maxiter': maxfun, 'ftol' : 1e-3,
                    'gtol': 10 ** (-self.nitem_features-self.nperson_features - 1)})

            opt_hyperparams = res['x']
            nlml = res['fun']
            nfits += res['nfev']

            if nlml < min_nlml:
                min_nlml = nlml
                best_opt_hyperparams = opt_hyperparams
                best_iter = r

            # choose a new lengthscale for the initial guess of the next attempt
            if r < nrestarts - 1:
                self.ls = gamma.rvs(self.shape_ls, scale=1.0 / self.rate_ls, size=len(self.ls))
                if person_features is not None:
                    self.lsy = gamma.rvs(self.shape_lsy, scale=1.0 / self.rate_lsy, size=len(self.lsy))

            final_r = r

        if best_iter < final_r:
            # need to go back to the best result
            if person_features is None:  # don't do this if further optimisation required anyway
                self.neg_marginal_likelihood(best_opt_hyperparams, 'item', -1, personIDs, items_1_coords, items_2_coords,
                                item_features, preferences, person_features, input_type=input_type,
                                use_median_ls=use_median_ls, use_MAP=False)

        logging.debug(
            "Chosen item length-scale %s, used %i evals of NLML over %i restarts" % (self.ls, nfits, nrestarts))
        if self.person_features is not None:
            logging.debug(
                "Chosen person length-scale %s, used %i evals of NLML over %i restarts" % (self.lsy, nfits, nrestarts))

        logging.debug("Optimal hyper-parameters: item = %s, person = %s" % (self.ls, self.lsy))
        return self.ls, self.lsy, -min_nlml  # return the log marginal likelihood

    def neg_marginal_likelihood(self, hyperparams, lstype, dimension, personIDs, items_1_coords, items_2_coords,
                                item_features, preferences, person_features, input_type,
                                use_median_ls, use_MAP=False):
        """
        Weight the marginal log data likelihood by the hyper-prior. Unnormalised posterior over the hyper-parameters.
        """
        if np.any(np.isnan(hyperparams)):
            return np.inf

        # set the correct hyperparameters according to lstype
        if lstype == 'item':
            if dimension == -1 or self.n_wlengthscales == 1:
                self.ls[:] = np.exp(hyperparams)
            else:
                self.ls[dimension] = np.exp(hyperparams)
            logging.debug("Testing item length-scale: %s" % (self.ls))

        elif lstype == 'person':
            if dimension == -1 or self.n_ylengthscales == 1:
                self.lsy[:] = np.exp(hyperparams)
            else:
                self.lsy[dimension] = np.exp(hyperparams)
            logging.debug("Testing person length-scale guess: %s" % (self.lsy))

        elif lstype == 'fa':
            new_Nfactors = int(np.round(np.exp(hyperparams)))

        elif lstype == 'both' and dimension <= 0:  # can be zero if single length scales or -1 to do all
            # person and item
            self.ls[:] = np.exp(hyperparams[:self.nitem_features])
            self.lsy[:] = np.exp(hyperparams[self.nitem_features:])
            logging.debug("Testing item length-scale: %s" % (self.ls))
            logging.debug("Testing person length-scale guess: %s" % (self.lsy))
        else:
            logging.error("Invalid length-scale type for optimization.")

        if np.any(np.isinf(self.ls)):
            return np.inf
        if np.any(np.isinf(self.lsy)):
            return np.inf

        self.fit(personIDs, items_1_coords, items_2_coords, item_features, preferences, person_features,
                 input_type=input_type, use_median_ls=use_median_ls)

        marginal_log_likelihood = self.lowerbound()
        if use_MAP:
            log_model_prior = self.ln_modelprior()
            lml = marginal_log_likelihood + log_model_prior
        else:
            lml = marginal_log_likelihood

        if lstype == 'person':
            if dimension == -1:
                logging.debug("LML: %f, %s length-scales = %s" % (lml, lstype, self.lsy))
            else:
                logging.debug(
                    "LML: %f, %s length-scale for dim %i = %.3f" % (lml, lstype, dimension, self.lsy[dimension]))
        elif lstype == 'item':
            if dimension == -1:
                logging.debug("LML: %f, %s length-scales = %s" % (lml, lstype, self.ls))
            else:
                logging.debug(
                    "LML: %f, %s length-scale for dim %i = %.3f" % (lml, lstype, dimension, self.ls[dimension]))
        elif lstype == 'both':
            logging.debug("LML: %f, item length-scales = %s, person length-scales = %s" % (lml, self.ls, self.lsy))
        return -lml

    def _gradient_dim(self, invK_mm, common_term, lstype, dimension):
        der_logpw_logqw = 0
        der_logpy_logqy = 0
        der_logpt_logqt = 0
        der_logpf_logqf = 0

        # compute the gradient. This should follow the MAP estimate from chu and ghahramani. 
        # Terms that don't involve the hyperparameter are zero; implicit dependencies drop out if we only calculate 
        # gradient when converged due to the coordinate ascent method.
        if lstype == 'item':

            dKdls = self.K * self.kernel_der(self.obs_coords, self.ls, dimension)
            invK_w = self.invK.dot(self.w)  # N x Nfactors

            for f in range(self.Nfactors):
                fidxs = np.arange(self.N) + (self.N * f)

                swf = self.shape_sw[f] / self.rate_sw[f]
                invKs_Cf = swf * self.invK.dot(self.w_cov[fidxs, :][:, fidxs])
                invK_wf = invK_w[:, f:f + 1]

                Sigma = self.Sigma_w[:, :, f]

                der_logpw_logqw += 0.5 * (invK_wf.T.dot(dKdls).dot(invK_wf) * swf -
                                          np.trace(invKs_Cf.dot(Sigma).dot(dKdls / swf)))

            if self.use_t:
                invK_t = self.invK.dot(self.t)
                invKs_C = self.st * self.invK.dot(self.t_cov)

                der_logpt_logqt = 0.5 * (invK_t.T.dot(dKdls).dot(invK_t) * self.st -
                                         np.trace(invKs_C.dot(self.t_gp.get_obs_precision()).dot(dKdls / self.st)))

        elif lstype == 'person' and self.person_features is not None:
            dKdls = self.Ky * self.kernel_der(self.person_features, self.lsy, dimension)
            invK_y = self.invKy.dot(self.y.T)  # Npeople x Nfactors

            for f in range(self.Nfactors):
                fidxs = np.arange(self.Npeople) + (self.Npeople * f)

                syf = self.shape_sy[f] / self.rate_sy[f]
                invKs_Cf = syf * self.invKy.dot(self.y_cov[fidxs, :][:, fidxs])
                invK_yf = invK_y[:, f:f + 1]

                Sigma = self.Sigma_y[:, :, f]

                der_logpy_logqy += 0.5 * (invK_yf.T.dot(dKdls).dot(invK_yf) * syf -
                                          np.trace((invKs_Cf.dot(Sigma)).dot(dKdls / syf)))

        return der_logpw_logqw + der_logpy_logqy + der_logpt_logqt + der_logpf_logqf

    def _compute_gradients_all_dims(self, lstype, dimensions):
        mll_jac = np.zeros(len(dimensions), dtype=float)

        if lstype == 'item' or (lstype == 'both'):
            common_term = np.sum(np.array([(self.w[:, f:f+1].dot(self.w[:, f:f+1].T) + self.w_cov[f]).dot(
                self.shape_sw[f] / self.rate_sw[f] * self.invK) - np.eye(self.N)
                for f in range(self.Nfactors)]), axis=0)
            if self.use_t:
                common_term += (self.t.dot(self.t.T) + self.t_cov).dot(self.shape_st / self.rate_st * self.invK) \
                               - np.eye(self.N)

            for dim in dimensions[:self.nitem_features]:
                mll_jac[dim] = self._gradient_dim(self.invK, common_term, 'item', dim)

        if (lstype == 'person' or (lstype == 'both')) and self.person_features is not None:
            common_term = np.sum(np.array([(self.y[f:f+1].T.dot(self.y[f:f+1,:]) + self.y_cov[f]).dot(self.invKy_block)
                                           - np.eye(self.Npeople) for f in range(self.Nfactors)]), axis=0)

            for dim in dimensions[self.nitem_features:]:
                mll_jac[dim + self.nitem_features] = self._gradient_dim(self.invKy_block, common_term, 'person', dim)

        return mll_jac

    def nml_jacobian(self, hyperparams, lstype, dimension, personIDs, items_1_coords, items_2_coords, item_features,
                     preferences, person_features, input_type, use_median_ls, use_MAP=False):
        """
        Weight the marginal log data likelihood by the hyper-prior. Unnormalised posterior over the hyper-parameters.
        """
        if np.any(np.isnan(hyperparams)):
            return np.inf

        needs_fitting = False

        if lstype == 'item':
            if dimension == -1 or self.n_wlengthscales == 1:
                if np.any(np.abs(self.ls - np.exp(hyperparams)) > 1e-4):
                    needs_fitting = True
                    self.ls[:] = np.exp(hyperparams)
                dimensions = np.arange(len(self.ls))
            else:
                if np.any(np.abs(self.ls[dimension] - np.exp(hyperparams)) > 1e-4):
                    needs_fitting = True
                    self.ls[dimension] = np.exp(hyperparams)
                dimensions = [dimension]

            lml_jac = np.zeros(len(dimensions))
            lml_jac[np.isneginf(self.ls)] = np.inf
            lml_jac[np.isposinf(self.ls)] = -np.inf
            if np.any(np.isinf(self.ls)):
                return lml_jac

        elif lstype == 'person':
            if dimension == -1 or self.n_ylengthscales == 1:
                if np.any(np.abs(self.lsy - np.exp(hyperparams)) > 1e-4):
                    needs_fitting = True
                    self.lsy[:] = np.exp(hyperparams)
                dimensions = np.arange(len(self.lsy))
            else:
                if np.any(np.abs(self.ls[dimension] - np.exp(hyperparams)) > 1e-4):
                    needs_fitting = True
                    self.lsy[dimension] = np.exp(hyperparams)
                dimensions = [dimension]

            lml_jac = np.zeros(len(dimensions))
            lml_jac[np.isneginf(self.lsy)] = np.inf
            lml_jac[np.isposinf(self.lsy)] = -np.inf
            if np.any(np.isinf(self.lsy)):
                return lml_jac

        elif lstype == 'both' and dimension <= 0:

            hyperparams_w = hyperparams[:self.nitem_features]
            hyperparams_y = hyperparams[self.nitem_features:]

            if np.any(np.abs(self.ls - np.exp(hyperparams_w)) > 1e-4):
                needs_fitting = True
                self.ls[:] = np.exp(hyperparams_w)

            if np.any(np.abs(self.lsy - np.exp(hyperparams_y)) > 1e-4):
                needs_fitting = True
                self.lsy[:] = np.exp(hyperparams_y)

            dimensions = np.append(np.arange(len(self.ls)), np.arange(len(self.lsy)))

            lml_jac = np.zeros(len(dimensions))
            ls_all = np.append(self.ls, self.lsy)
            lml_jac[np.isneginf(ls_all)] = np.inf
            lml_jac[np.isposinf(ls_all)] = -np.inf
            if np.any(np.isinf(ls_all)):
                return lml_jac
        else:
            logging.error("Invalid optimization setup.")

        if needs_fitting:
            self.neg_marginal_likelihood(hyperparams, lstype, dimension, personIDs, items_1_coords, items_2_coords,
                    item_features, preferences, person_features, input_type, use_median_ls, use_MAP)

        # num_jobs = multiprocessing.cpu_count()
        # if num_jobs > 12:
        #     num_jobs = 12

        if self.verbose:
            logging.debug("Starting gradient computations...")
        # mll_jac = Parallel(n_jobs=num_jobs)(delayed(self._gradient_dim)(lstype, d, dim)
        #                                      for d, dim in enumerate(dimensions))
        # mll_jac = np.array(mll_jac)  #, order='F')
        mll_jac = self._compute_gradients_all_dims(lstype, dimensions)

        if self.verbose:
            logging.debug("Completed gradient computations.")

        if len(mll_jac) == 1:  # don't need an array if we only compute for one dimension
            mll_jac = mll_jac[0]
        elif (lstype == 'item' and self.n_wlengthscales == 1) or (lstype == 'person' and self.n_ylengthscales == 1):
            mll_jac = np.sum(mll_jac)
        elif lstype == 'both':
            if self.n_wlengthscales == 1:
                mll_jac[:self.nitem_features] = np.sum(mll_jac[:self.nitem_features])
            if self.n_ylengthscales == 1:
                mll_jac[self.nitem_features:] = np.sum(mll_jac[self.nitem_features:])

        if use_MAP:  # gradient of the log prior
            log_model_prior_grad = self.ln_modelprior_grad()
            lml_jac = mll_jac + log_model_prior_grad
        else:
            lml_jac = mll_jac
        logging.debug("Jacobian of LML: %s" % lml_jac)
        if self.verbose:
            logging.debug("...with item length-scales = %s, person length-scales = %s" % (self.ls, self.lsy))
        return lml_jac

    def ln_modelprior(self):
        # Gamma distribution over each value. Set the parameters of the gammas.
        lnp_gp = - gammaln(self.shape_ls) + self.shape_ls * np.log(self.rate_ls) \
                 + (self.shape_ls - 1) * np.log(self.ls) - self.ls * self.rate_ls

        lnp_gpy = - gammaln(self.shape_lsy) + self.shape_lsy * np.log(self.rate_lsy) \
                  + (self.shape_lsy - 1) * np.log(self.lsy) - self.lsy * self.rate_lsy

        return np.sum(lnp_gp) + np.sum(lnp_gpy)

        # UTILITY FUNCTIONS ----------------------------------------------------------------------------------------------

    def pickle_me(self, filename):
        """
        Save the object as a pickle -- avoids errors that occur if you simply try to dump an instance of this class.
        :param filename:
        :return:
        """
        import pickle
        from copy import deepcopy
        with open(filename, 'wb') as fh:
            m2 = deepcopy(self)
            if hasattr(m2, 't_gp'):
                m2.t_gp.kernel_func = None
            m2.wy_gp.kernel_func = None  # have to do this to be able to pickle

            pickle.dump(m2, fh)

        #
# class PreferenceNoComponentFactors(PreferenceComponents):
#     """
#     Class for preference learning with multiple users, where each user has a GP. No sharing of information between users
#     and no latent components.
#     """
#     def __init__(self, nitem_features, nperson_features=0, mu0=0, shape_s0=1, rate_s0=1,
#                  shape_ls=1, rate_ls=100, ls=100, shape_lsy=1, rate_lsy=100, lsy=100, verbose=False, nfactors=20,
#                  use_common_mean_t=True, kernel_func='matern_3_2',
#                  max_update_size=10000, ninducing=500, forgetting_rate=0.9, delay=1.0):
#
#
#         PreferenceComponents.__init__(self, nitem_features, nperson_features, mu0, shape_s0, rate_s0,
#                  shape_ls, rate_ls, ls, shape_lsy, rate_lsy, lsy, verbose, nfactors,
#                  use_common_mean_t, kernel_func,
#                  max_update_size, ninducing, forgetting_rate, delay)
#         self.use_t = False
#
#     def _init_w(self):
#         self.w = np.zeros((self.N, self.Nfactors))
#
#     def _init_y(self):
#         self.y = np.zeros((self.Nfactors, self.Npeople))
#
#     def _predict_w_t(self, coords_1):
#         # kernel between pidxs and t
#         w1 = np.zeros((coords_1.shape[0], self.Nfactors))
#         t1 = np.zeros((coords_1.shape[0], 1))
#
#         return t1, w1
#
#     def _predict_y(self, _, Npeople):
#         return super(PreferenceNoComponentFactors, self)._predict_y(None, Npeople)
#
#     def _gradient_dim(self, lstype, d, dimension):
#         der_logpf_logqf = 0
#
#         if lstype == 'item' or (lstype == 'both' and d < self.nitem_features):
#             for p in self.pref_gp:
#                 der_logpf_logqf += self.pref_gp[p].lowerbound_gradient(dimension)
#
#         return der_logpf_logqf
#
#     def _expec_f_p(self, p, mu0_output):
#         f, _ = self.pref_gp[p].predict_f(
#                 out_idxs=self.coordidxs[p] if self.vb_iter==0 else None,
#                 out_feats=self.obs_coords if self.vb_iter==0 else None,
#                 mu0_output=mu0_output,
#                 reuse_output_kernel=True)
#         self.f[self.coordidxs[p], p] = f.flatten()
#
#     def _expec_w(self):
#         return
#
#     def lowerbound(self):
#         f_terms = 0
#
#         for p in self.pref_gp:
#             f_terms += self.pref_gp[p].lowerbound()
#             if self.verbose:
#                 logging.debug('s_f^%i=%.2f' % (p, self.pref_gp[p].s))
#
#         lb = f_terms
#
#         if self.verbose:
#             logging.debug( "Iteration %i: Lower bound = %.3f, " % (self.vb_iter, lb) )
#
#         return lb
#
# class PreferenceComponentsFA(PreferenceComponents):
#     # Factor Analysis
#     def _init_w(self):
#         self.fa = FactorAnalysis(n_components=self.Nfactors)
#         self.w = np.zeros((self.N, self.Nfactors))
#
#     def _init_y(self):
#         self.y = np.ones((self.Nfactors, self.Npeople))
#
#     def _optimize(self, personIDs, items_1_coords, items_2_coords, item_features, preferences, person_features=None,
#                  maxfun=20, use_MAP=False, nrestarts=1, input_type='binary'):
#
#         max_iter = self.max_iter
#         self.fit(personIDs, items_1_coords, items_2_coords, item_features, preferences, person_features, input_type=input_type)
#         self.max_iter = max_iter
#
#         min_nlml = np.inf
#         best_opt_hyperparams = None
#         best_iter = -1
#
#         logging.debug("Optimising item length-scale for all dimensions")
#
#         nfits = 0 # number of calls to fit function
#
#         # optimise each length-scale sequentially in turn
#         for r in range(nrestarts):
#             # try to do it using the conjugate gradient method instead. Requires Jacobian (gradient) of LML
#             # approximation. If we also have Hessian or Hessian x arbitrary vector p, we can use Newton-CG, dogleg,
#             # or trust-ncg, which may be faster still?
#             initialguess = np.log(self.ls)
#             logging.debug("Initial item length-scale guess in restart %i: %s" % (r, self.ls))
#             res = minimize(self.neg_marginal_likelihood, initialguess, args=('item', -1, use_MAP,),
#                    jac=self.nml_jacobian, method='L-BFGS-B', options={'maxiter':maxfun, 'gtol': 0.1 / self.nitem_features})
#
#             opt_hyperparams = res['x']
#             nlml = res['fun']
#             nfits += res['nfev']
#
#             if nlml < min_nlml:
#                 min_nlml = nlml
#                 best_opt_hyperparams = opt_hyperparams
#                 best_iter = r
#
#             # choose a new lengthscale for the initial guess of the next attempt
#             if r < nrestarts - 1:
#                 self.ls = gamma.rvs(self.shape_ls, scale=1.0/self.rate_ls, size=len(self.ls))
#
#         if best_iter < r:
#             # need to go back to the best result
#             self.neg_marginal_likelihood(best_opt_hyperparams, 'item', -1, use_MAP=False)
#
#         logging.debug("Chosen item length-scale %s, used %i evals of NLML over %i restarts" % (self.ls, nfits, nrestarts))
#
#         initialguess = np.log(self.Nfactors)
#         res = minimize(self.neg_marginal_likelihood, initialguess, args=('fa', -1, use_MAP,),
#                method='Nelder-Mead', options={'maxfev':maxfun, 'xatol':np.mean(self.ls) * 1e100, 'return_all':True})
#         min_nlml = res['fun']
#         logging.debug("Optimal number of factors = %s, with initialguess=%i and %i function evals" % (self.Nfactors,
#                                                                        int(np.exp(initialguess)), res['nfev']))
#
#         logging.debug("Optimal hyper-parameters: item = %s" % (self.ls))
#         return self.ls, self.lsy, -min_nlml
#
#     def _gradient_dim(self, lstype, d, dimension):
#         der_logpw_logqw = 0
#         der_logpy_logqy = 0
#         der_logpt_logqt = 0
#         der_logpf_logqf = 0
#
#         # compute the gradient. This should follow the MAP estimate from chu and ghahramani.
#         # Terms that don't involve the hyperparameter are zero; implicit dependencies drop out if we only calculate
#         # gradient when converged due to the coordinate ascent method.
#         if lstype == 'item' or (lstype == 'both' and d < self.nitem_features):
#             for p in self.pref_gp:
#                 der_logpf_logqf += self.pref_gp[p].lowerbound_gradient(dimension)
#
#         elif lstype == 'person' or (lstype == 'both' and d >= self.nitem_features):
#             return 0
#
#         return der_logpw_logqw + der_logpy_logqy + der_logpt_logqt + der_logpf_logqf
#
#     def _predict_w_t(self, coords_1):
#         if self.cov_type == 'matern_3_2':
#             kernel = Matern(self.ls)
#         else:
#             logging.error('Kernel not implemented for FA')
#             return 0
#
#         w1 = np.zeros((coords_1.shape[0], self.Nfactors))
#         for f in range(self.Nfactors):
#             w_gp = GPR(kernel=kernel, optimizer=None)
#             w_gp.fit(self.obs_coords, self.w[:, f])
#             w1[:, f] = w_gp.predict(coords_1, return_std=False)
#
#         w_gp = GPR(kernel, optimizer=None)
#         w_gp.fit(self.obs_coords, self.t)
#         t1 = w_gp.predict(coords_1, return_std=False)
#
#         return t1, w1, np.zeros((self.N, self.N)), np.zeros((self.N*self.Nfactors, self.N*self.Nfactors))
#
#     def _predict_y(self, person_features, Npeople):
#
#         y1 = np.zeros((self.Nfactors, Npeople))
#         if person_features is None:
#             return y1
#
#         if self.cov_type == 'matern_3_2':
#             kernel = Matern(self.ls)
#         else:
#             logging.error('Kernel not implemented for FA')
#             return 0
#
#         for f in range(self.Nfactors):
#             y_gp = GPR(kernel=kernel, optimizer=None)
#             y_gp.fit(self.person_features, self.y[f, :])
#             y1[f, :] = y_gp.predict(person_features, return_std=False)
#
#         return y1, np.zeros((self.Npeople, self.Npeople))
#
#     def _expec_w(self):
#         """
#         Compute the expectation over the latent features of the items and the latent personality components
#         """
#         self.y = self.fa.fit_transform(self.f.T).T
#         self.w = self.fa.components_.T
#         self.wy = self.w.dot(self.y)
#         return
#
#     def _expec_t(self):
#         self.t = self.fa.mean_[:, np.newaxis]
#         return
#
#     def lowerbound(self):
#         f_terms = 0
#
#         for p in self.pref_gp:
#             f_terms += self.pref_gp[p].lowerbound()
#             if self.verbose:
#                 logging.debug('s_f^%i=%.2f' % (p, self.pref_gp[p].s))
#
#         lb = np.sum(self.fa.score_samples(self.f.T)) + f_terms
#         if self.verbose:
#             logging.debug( "Iteration %i: Lower bound = %.3f, " % (self.vb_iter, lb) )
#         return lb
#
