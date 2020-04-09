# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse import coo_matrix, issparse
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.special import gammaln, psi, binom
from scipy.stats import gamma
import logging
from joblib import Parallel, delayed
import multiprocessing

max_no_jobs = 2

def compute_distance(col, row):
    # create a grid where each element of the row is subtracted from each element of the column
    if issparse(col) or issparse(row):
        # col = col[:, np.zeros(row.shape[1])]
        # row = row[np.zeros(col.shape[0]), :]
        col = col.A  # we assume the column and row are taken from a larger sparse matrix so converting to nparray is ok
        row = row.A
    # else: # this is trivial for a numpy array
    return col - row


def coord_arr_to_1d(arr):
    arr = np.ascontiguousarray(arr)
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))


def coord_arr_from_1d(arr, dtype, dims):
    arr = arr.view(dtype)
    return arr.reshape(dims)


def sigmoid(f):
    g = 1 / (1 + np.exp(-f))
    return g


def logit(g):
    f = -np.log(1 / g - 1)
    return f


def target_var(f, s, v):
    mean = sigmoid(f, s)
    u = mean * (1 - mean)
    v = v * s * s
    return u / (1 / (u * v) + 1)


def temper_extreme_probs(probs, zero_only=False):
    if not zero_only:
        probs[probs > 1 - 1e-7] = 1 - 1e-7

    probs[probs < 1e-7] = 1e-7

    return probs


# Kernels

# Diagonal

def derivfactor_diag_from_raw_vals(vals, vals2, ls_d, operator='*'):
    return 0


def diagonal_from_raw_vals(vals, ls, vals2=None, operator='*', vector=False):
    '''
    No covariance between different locations. Operator is ignored.
    '''
    if vals2 is None:
        vals2 = vals

    if vector:
        K = np.zeros(vals.shape[0], dtype=float)
        for i in range(vals.shape[0]):
            K[i] = float(np.all(vals[i] == vals2[i]))

        return K

    K = np.zeros((vals.shape[0], vals2.shape[0]), dtype=float)
    for i in range(vals.shape[0]):
        # check for the same locations.
        K[i, :] = (np.sum(np.abs(vals[i:i + 1, :] - vals2), axis=1) == 0).astype(float)
    return K


# Squared Exponential -- needs converting to work with raw values

# def sq_exp_cov(distances, ls):
#     K = np.zeros(distances.shape)
#     for d in range(distances.shape[2]):
#         K[:, :, d] = np.exp( -distances[:, :, d]**2 / ls[d] )
#     K = np.prod(K, axis=2)
#     return K
#
# def deriv_sq_exp_cov(distances, ls, dim):
#     K_notdim = np.ones(distances.shape)
#     for d in range(distances.shape[2]):
#         if d == dim:
#             K_dim = np.exp( -distances[:, :, d]**2 / ls[d] ) * -distances[:, :, d]**2 / ls[d]**2
#             continue
#         K_notdim[:, :, d] = np.exp( -distances[:, :, d]**2 / ls[d] )
#     K_notdim = np.prod(K_notdim, axis=2)
#     return K_notdim * K_dim

# Matern 3/2

def matern_3_2_onedimension_from_raw_vals(xvals, x2vals, ls_d, vector=False):
    xvals = xvals * 3 ** 0.5 / ls_d
    x2vals = x2vals * 3 ** 0.5 / ls_d

    if vector:
        K = compute_distance(xvals, x2vals)
    else:
        K = compute_distance(xvals, x2vals.T)
    K = np.abs(K, K)

    exp_minusK = np.exp(-K)
    K *= exp_minusK
    K += exp_minusK

    return K


def derivfactor_matern_3_2_from_raw_vals_onedimension(vals, vals2, ls_d, operator='*'):
    '''
    To obtain the derivative W.R.T the length scale indicated by dim, multiply the value returned by this function
    with the kernel. Use this to save recomputing the kernel for each dimension.
    '''
    if ls_d == 0:
        dKdls_d = np.inf
        return dKdls_d

    D = np.abs(compute_distance(vals, vals2.T))
    # K = -(1 + K) * (ls_d - K) * np.exp(-K / ls_d) / ls_d**3
    sqrt3d = D * 3 ** 0.5 / ls_d
    exp_minus_sqrt3d = np.exp(-sqrt3d)
    dKdls_d = 3 * D ** 2 * exp_minus_sqrt3d / ls_d ** 3

    if operator == '*':
        Kfactor = sqrt3d * exp_minus_sqrt3d
        Kfactor += exp_minus_sqrt3d

        Kfactor[Kfactor == 0] = np.nextafter(0,1)

        dKdls_d /= Kfactor

    return dKdls_d


def derivfactor_matern_3_2_from_raw_vals(vals, ls, d, vals2=None, operator='*'):
    '''
    To obtain the derivative W.R.T the length scale indicated by dim, multiply the value returned by this function
    with the kernel. Use this to save recomputing the kernel for each dimension.
    '''
    if len(ls) > 1:
        ls_d = ls[d]
    else:
        ls_d = ls[0]

    xvals = vals[:, d:d + 1]
    if vals2 is None:
        xvals2 = xvals
    else:
        xvals2 = vals2[:, d:d + 1]

    D = np.abs(compute_distance(xvals, xvals2.T))
    # K = -(1 + K) * (ls_d - K) * np.exp(-K / ls_d) / ls_d**3
    K = 3 * D ** 2 * np.exp(-D * 3 ** 0.5 / ls_d) / ls_d ** 3

    if len(ls) > 1:
        ls_i = ls[d]
    else:
        ls_i = ls[0]

    if operator == '*':
        Kfactor_i = matern_3_2_onedimension_from_raw_vals(xvals, xvals2, ls_i)
        K[K != 0] = K[K!=0] / Kfactor_i[K!=0]
    elif operator == '+':
        K /= float(vals.shape[1])

    return K


def matern_3_2_from_raw_vals(vals, ls, vals2=None, operator='*', n_threads=0, vector=False):

    # if n_threads == 0:
    #     num_jobs = multiprocessing.cpu_count()
    #     if num_jobs > max_no_jobs:
    #         num_jobs = max_no_jobs
    # else:
    #     num_jobs = n_threads
    #
    # subset_size = int(np.ceil(vals.shape[1] / float(num_jobs)))
    # K = Parallel(n_jobs=num_jobs, backend='threading')(delayed(compute_K_subset)(i, subset_size, vals, vals2, ls,
    #                                                                          matern_3_2_onedimension_from_raw_vals,
    #                                                                          operator, vector) for i in range(num_jobs))

    if vals2 is None:
        dists = pdist((vals/ls), metric='euclidean')
    elif vector:
        dists = np.sqrt(np.sum((vals/ls) ** 2 + (vals2/ls) ** 2 - 2 * (vals/ls) * (vals2/ls), axis=1))
    else:
        dists = cdist((vals/ls), (vals2/ls), metric='euclidean')

    K = dists * np.sqrt(3)
    K = (1. + K) * np.exp(-K)

    if vals2 is None and not vector:
        # convert from upper-triangular matrix to square matrix
        K = squareform(K)
        np.fill_diagonal(K, 1)

    return K


def compute_K_subset(subset, subset_size, vals, vals2, ls, fun, operator, vector=False):
    if operator == '*':
        K_subset = 1
    elif operator == '+':
        K_subset = 0

    range_end = subset_size * (subset + 1)
    if range_end > vals.shape[1]:
        range_end = vals.shape[1]

    for i in range(subset_size * subset, range_end):
        if np.mod(i, 10000) == 9999:
            logging.debug("computing kernel for feature %i" % i)
        xvals = vals[:, i:i + 1]

        if vals2 is None:
            xvals2 = xvals
        else:
            xvals2 = vals2[:, i:i + 1]

        if len(ls) > 1:
            ls_i = ls[i]
        else:
            ls_i = ls[0]
        K_d = fun(xvals, xvals2, ls_i, vector)
        if operator == '*':
            K_subset *= K_d
        elif operator == '+':
            K_subset += K_d
    return K_subset


def _dists_f(items_feat_sample, f):
    if np.mod(f, 1000) == 0:
        logging.debug('computed lengthscale for feature %i' % f)
    dists = np.abs(items_feat_sample[:, np.newaxis] - items_feat_sample[np.newaxis, :])
    # we exclude the zero distances. With sparse features, these would likely downplay the lengthscale.
    if np.any(dists > 0):
        med = np.median(dists[dists > 0])
    else:
        med = 1.0

    if np.isnan(med) or med == 0:
        med = 1.0
    return med


def compute_median_lengthscales(items_feat, N_max=3000, n_threads=0, multiply_heuristic_power=0.5):
    if items_feat.shape[0] > N_max:
        items_feat = items_feat[np.random.choice(items_feat.shape[0], N_max, replace=False)]

    ndims = items_feat.shape[1]

    if n_threads == 0:
        num_jobs = multiprocessing.cpu_count()
        if num_jobs > max_no_jobs:
            num_jobs = max_no_jobs
    else:
        num_jobs = n_threads

    logging.debug('Creating %i jobs for the lengthscales' % num_jobs)

    default_ls_value = Parallel(n_jobs=num_jobs, backend="threading")(delayed(_dists_f)(
        items_feat[:, f], f) for f in range(ndims))

    ls_initial_guess = np.ones(ndims) * default_ls_value

    if items_feat.shape[1] > 200:
        ls_initial_guess *= items_feat.shape[1] ** multiply_heuristic_power

    # this is a heuristic, see e.g. "On the High-dimensional
    # Power of Linear-time Kernel Two-Sample Testing under Mean-difference Alternatives" by Ramdas et al. 2014. In that
    # paper they refer to root(no. dimensions) because they square the lengthscale in the kernel function.
    # It's possible that this value should be higher if a lot of feature values are actually missing values, as these
    # would lower the median.
    # If multiply_heuristic_power is 0.5, we are normalising the total euclidean distance by number of dimensions (
    # think Pythagoras' theorem).

    return ls_initial_guess

def check_convergence(newval, oldval, conv_threshold, positive_only, iter=-1, verbose=False, label='', change_as_a_fraction=True):
    """
    Test whether a method has converged given values of interest and threshold.
    Assumes convergence if the difference between the two values as a fraction of the new value is below
    the threshold.  Use this if the magnitude of the values is not known a priori, so that convergence is assumed
    when the change is below a certain fraction of the values.

    :param newval: new value of variable we test for convergence. May be a vector to test multiple variables
    :param oldval: previous value of variable we are testing for convergence. May be a vector if we want to test multiple variables.
    :param conv_threshold: threshold below which the fractional difference must fall before convergence is declared.
    :param positive_only: if True, assumes that values must simply be below the threshold, and that negative differences are bad and prints warnings if they occur.
    :param iter: iteration number, used to print debugging logs. Set to -1 to turn off the logging
    :param label: method name or other label used to identify the output from this test in debugging logs
    :return: True or False depending on whether the fraction is below the threshold
    """

    diff = newval - oldval

    if change_as_a_fraction:
        diff /= np.abs(newval)

    if not positive_only:
        diff = np.abs(diff)

    # if we are testing a vector of multiple variables, consider the biggest difference
    diff = np.max(diff)

    #if verbose:
    if np.isscalar(newval):
        logging.debug('%s: %.5f, diff = %f at iteration %i' % (label, newval, diff, iter))
    elif newval.size == 1:
        logging.debug('%s: %.5f, diff = %f at iteration %i' % (label, newval.flatten()[0], diff, iter))
    else:
        logging.debug('%s: diff = %f at iteration %i' % (label, diff, iter))
            # logging.debug(np.max(np.abs(newval - oldval)))

    converged = diff < conv_threshold

    if positive_only and diff < - 10000 * conv_threshold:  # ignore any small errors as we are using approximations
        logging.warning('%s = %.5f, changed by %.5f (converged = %i) in iteration %i' %
                        (label, newval, diff, converged, iter))

    return converged

class GPClassifierVB(object):
    verbose = False

    # hyper-parameters
    s = 1  # inverse output scale
    ls = 100  # inner length scale of the GP
    n_lengthscales = 1  # separate or single length scale?
    mu0 = 0

    # parameters for the hyper-priors if we want to optimise the hyper-parameters
    shape_ls = 1
    rate_ls = 0
    shape_s0 = 1
    rate_s0 = 1

    # save the training points
    obs_values = None  # value of the positive class at each observations. Any duplicate points will be summed.
    obs_f = None
    obs_C = None

    out_feats = None

    G = None
    z = None
    K = None
    Q = None
    f = None
    v = None
    K_star = None

    n_converged = 1  # number of iterations while the algorithm appears to be converged -- in case of local maxima
    max_iter_VB_per_fit = 200  # 1000
    min_iter_VB = 5
    max_iter_G = 10
    conv_threshold = 1e-4
    conv_threshold_G = 1e-5
    conv_check_freq = 2

    uselowerbound = True

    p_rep = 1.0  # weight report values by a constant probability to indicate uncertainty in the reports

    def __init__(self, ninput_features, z0=0.5, shape_s0=2, rate_s0=2, shape_ls=10, rate_ls=1, ls_initial=None,
                 kernel_func='matern_3_2', kernel_combination='*', verbose=False, fixed_s=False):

        self.verbose = verbose

        self.n_locs = 0 # at this point we have no training locations
        self.max_iter_VB = self.max_iter_VB_per_fit

        # Grid size for prediction
        self.ninput_features = ninput_features

        if ls_initial is not None:
            self.n_lengthscales = len(ls_initial)  # can pass in a single length scale to be used for all dimensions
        else:
            self.n_lengthscales = self.ninput_features

        # Output scale (sigmoid scaling)
        self.shape_s0 = float(shape_s0)  # prior pseudo counts * 0.5
        self.rate_s0 = float(rate_s0)  # prior sum of squared deviations
        self.shape_s = float(shape_s0)
        self.rate_s = float(rate_s0)  # init to the priors until we receive data
        self.s = self.shape_s0 / self.rate_s0

        self.fixed_s = fixed_s

        # Prior mean
        self._init_prior_mean_f(z0)

        # Length-scale
        self.shape_ls = shape_ls  # prior pseudo counts * 0.5
        self.rate_ls = rate_ls  # analogous to a sum of changes between points at a distance of 1
        if np.any(ls_initial) and len(ls_initial) > 1:
            self.ls = np.array(ls_initial)
        else:
            self.ls = self.shape_ls / self.rate_ls
            if np.any(ls_initial):
                self.ls = np.zeros(self.ninput_features) + ls_initial[0]
            else:
                self.ls = np.zeros(self.ninput_features) + self.ls

        self.ls = self.ls.astype(float)

        # Algorithm configuration
        self._select_covariance_function(kernel_func)
        self.kernel_combination = kernel_combination  #  operator for combining kernels for each feature.

        self.vb_iter = 0

        self.features = None # an optional matrix of object feature vectors. Is not needed if coordinates are passed in

        self.n_threads = 0 # maximum number of threads, or zero if you want to use as many as possible

    def set_max_threads(self, n_threads):
        self.n_threads = n_threads

    # Initialisation --------------------------------------------------------------------------------------------------

    def _init_params(self, mu0, reinit_params, K=None):
        self._init_obs_mu0(mu0)

        if reinit_params or K is not None:
            self.K = K
            self._init_covariance()

        # Prior noise variance
        self.estimate_obs_noise()

        if reinit_params:

            self._init_obs_f()
            self._init_s()
            # g_obs_f = self._update_jacobian(G_update_rate) # don't do this here otherwise loop below will repeat the
            # same calculation with the same values, meaning that the convergence check will think nothing changes in the
            # first iteration.
            if self.G is None:
                self.G = 0

    def _init_covariance(self):
        # Get the correct covariance matrix
        if self.kernel_func is not None and self.cov_type != 'pre':
            self.K = self.kernel_func(self.obs_coords, self.ls, operator=self.kernel_combination, n_threads=self.n_threads)
            self.K += 1e-6 * np.eye(len(self.K))  # jitter
        elif self.K is None:
            logging.error('With covariance type "pre", the kernel must be passed in when calling fit()')

        self.invK = np.linalg.inv(self.K)

        self.Ks = self.K / self.s
        self.obs_C = self.K / self.s

        # Initialise here to speed up dot product -- assume we need to do this whenever there is new data
        self.Cov = np.zeros((self.Ntrain, self.Ntrain))
        self.KsG = np.zeros((self.n_locs, self.Ntrain))

    def reset_kernel(self):
        self.K = None
        self.invK = None
        self.Ks = None
        self.obs_C = None
        self.Cov = None
        self.KsG = None

    def _init_prior_mean_f(self, z0):
        self.mu0_default = logit(z0)

    def _init_obs_mu0(self, mu0):
        if mu0 is None:
            mu0 = self.mu0_default

        self.mu0_input = mu0 # save because in some cases we pass in a scalar and it is more convenient to work with this
        self.mu0 = np.zeros((self.n_locs, 1)) + mu0
        self.Ntrain = self.obs_values.size

    def _init_obs_f(self):
        # Mean probability at observed points given local observations
        if self.obs_f is None: # don't reset if we are only adding more data
            self.obs_f = logit(self.obs_mean)
        elif self.obs_f.shape[0] < self.obs_mean.shape[0]:
            prev_obs_f = self.obs_f
            self.obs_f = logit(self.obs_mean)
            self.obs_f[:prev_obs_f.shape[0], :] = prev_obs_f

    def estimate_obs_noise(self):
        """

        :param mu0: pass in the original mu0 here so we can use more efficient computation if it is scalar
        :return:
        """
        self._init_obs_prior()

        if not len(self.nu0):
            self.Q = []
            return

        # Noise in observations
        nu0_total = np.sum(self.nu0, axis=0)
        self.obs_mean = (self.obs_values + self.nu0[1]) / (self.obs_total_counts + nu0_total)

        # if we factorize the likelihood then  take expectation of the log likelihood with respect to f
        #obs_var = (self.obs_mean * (1 - self.obs_mean)) / (self.obs_total_counts + nu0_total + 1)
        self.Q = (self.obs_mean * (1 - self.obs_mean) ) / self.obs_total_counts
        self.Q = self.Q.flatten()

    def _init_obs_prior(self):

        mu0 = self.mu0_input
        if np.isscalar(mu0):
            n_locs = 1 # sample only once, and use the estimated values across all points
        else:
            n_locs = len(mu0)

        f_samples = np.random.normal(loc=mu0, scale=np.sqrt(self.rate_s0 / self.shape_s0),
                                     size=(n_locs, 1000))
        rho_samples = self.forward_model(f_samples)
        rho_mean = np.mean(rho_samples)
        rho_var = np.var(rho_samples)
        # find the beta parameters
        a_plus_b = 1.0 / (rho_var / (rho_mean * (1 - rho_mean))) - 1
        a = a_plus_b * rho_mean
        b = a_plus_b * (1 - rho_mean)
        self.nu0 = np.array([b, a])
        # if self.verbose:
        #    logging.debug("Prior parameters for the observation noise variance are: %s" % str(self.nu0))

    def _init_s(self):
        if not self.fixed_s:
            self.shape_s = self.shape_s0 + self.n_locs / 2.0
            self.rate_s = self.rate_s0 + 0.5 * (np.sum((self.obs_f-self.mu0)**2) + self.n_locs*self.rate_s0/self.shape_s0)
        self.s = self.shape_s / self.rate_s
        self.Elns = psi(self.shape_s) - np.log(self.rate_s)
        self.old_s = self.s
        if self.verbose:
            logging.debug("Setting the initial precision scale to s=%.3f" % self.s)

    def _select_covariance_function(self, cov_type):
        self.cov_type = cov_type

        if cov_type == 'matern_3_2':
            self.kernel_func = matern_3_2_from_raw_vals
            self.kernel_derfactor = derivfactor_matern_3_2_from_raw_vals_onedimension

        elif cov_type == 'diagonal':
            self.kernel_func = diagonal_from_raw_vals
            self.kernel_derfactor = derivfactor_diag_from_raw_vals

        #         elif cov_type == 'sq_exp': # no longer works -- needs kernel functions that work with the raw values
        #             self.kernel_func = sq_exp_cov
        #             self.kernel_der = deriv_sq_exp_cov

        elif cov_type is 'pre':
            # the covariance matrix is prespecified and passed in using set_covariance()
            self.kernel_func = None
            self.kernel_derfactor = None
        else:
            logging.error('GPClassifierVB: Invalid covariance type %s' % cov_type)

    # Input data handling ---------------------------------------------------------------------------------------------

    def _count_observations(self, obs_coords, n_obs, poscounts, totals):
        obs_coords = np.array(obs_coords)
        if obs_coords.shape[0] == self.ninput_features and obs_coords.shape[1] != self.ninput_features:
            if obs_coords.ndim == 3 and obs_coords.shape[2] == 1:
                obs_coords = obs_coords.reshape((obs_coords.shape[0], obs_coords.shape[1]))
            obs_coords = obs_coords.T

        if self.features is not None:
            self.obs_uidxs = np.arange(self.features.shape[0])
            # poscounts = poscounts.astype(int)
            totals = totals.astype(int)
            self.obs_coords = self.features
            self.n_obs = self.obs_coords.shape[0]
            return poscounts, totals

        # duplicate locations should be merged and the number of duplicates counted
        ravelled_coords = coord_arr_to_1d(obs_coords)
        uravelled_coords, origidxs, idxs = np.unique(ravelled_coords, return_index=True, return_inverse=True)

        grid_obs_counts = coo_matrix((totals, (idxs, np.ones(n_obs, dtype=int)))).toarray()
        grid_obs_pos_counts = coo_matrix((poscounts, (idxs, np.ones(n_obs, dtype=int)))).toarray()
        nonzero_idxs = grid_obs_counts.nonzero()[0]  # ravelled coordinates with duplicates removed
        uravelled_coords_nonzero = uravelled_coords[nonzero_idxs]
        origidxs_nonzero = origidxs[nonzero_idxs]
        # preserve the original order
        sortedidxs = np.argsort(origidxs_nonzero)
        uravelled_coords_nonzero_sorted = uravelled_coords_nonzero[sortedidxs]
        self.obs_coords = coord_arr_from_1d(uravelled_coords_nonzero_sorted, obs_coords.dtype,
                                            [nonzero_idxs.size, self.ninput_features])
        self.obs_uidxs = origidxs_nonzero[
            sortedidxs]  # records the mapping from the input data to the obs_coords object
        self.n_obs = self.obs_coords.shape[0]

        pos_counts = grid_obs_pos_counts[nonzero_idxs, 1][sortedidxs]
        totals = grid_obs_counts[nonzero_idxs, 1][sortedidxs]

        return pos_counts, totals

    def _process_observations(self, obs_coords, obs_values, totals=None):
        if obs_values is None:
            return [], []

        obs_values = np.array(obs_values)
        n_obs = obs_values.shape[0]

        if self.verbose:
            logging.debug("GP inference with %i observed data points." % n_obs)

        if totals is None or not np.any(totals >= 0):
            if (obs_values.ndim == 1 or obs_values.shape[
                1] == 1):  # obs_value is one column with values of either 0 or 1
                totals = np.ones(n_obs, dtype=int)
            else:  # obs_values given as two columns: first is positive counts, second is total counts.
                totals = obs_values[:, 1]
        elif (obs_values.ndim > 1) and (obs_values.shape[1] == 2):
            logging.warning(
                'GPClassifierVB received two sets of totals; ignoring the second column of the obs_values argument')

        if (obs_values.ndim == 1 or obs_values.shape[1] == 1):  # obs_value is one column with values of either 0 or 1
            poscounts = obs_values.flatten()
        elif (obs_values.shape[
                  1] == 2):  # obs_values given as two columns: first is positive counts, second is total counts.
            poscounts = obs_values[:, 0]

        if np.any(obs_values >= 0):
            poscounts[poscounts == 1] = self.p_rep
            poscounts[poscounts == 0] = 1 - self.p_rep

        # remove duplicates etc.
        poscounts, totals = self._count_observations(obs_coords, n_obs, poscounts, totals)
        self.obs_values = poscounts[:, np.newaxis].astype(float)
        self.obs_total_counts = totals[:, np.newaxis]
        n_locations = self.obs_coords.shape[0]
        self.n_locs = n_locations

        if self.verbose:
            logging.debug("Number of observed locations =" + str(self.obs_values.shape[0]))

        self._observations_to_z()

        self.K_out = {}  # store the output cov matrix for each block. Reset when we have new observations.
        self.K_star_diag = {}

    def _observations_to_z(self):
        obs_probs = self.obs_values / self.obs_total_counts
        self.z = obs_probs

    # Mapping between latent and observation spaces -------------------------------------------------------------------

    def forward_model(self, f, subset_idxs=None):
        if subset_idxs is not None:
            return sigmoid(f[subset_idxs])
        else:
            return sigmoid(f)

    def _compute_jacobian(self, f=None):

        if f is None:
            f = self.obs_f

        g_obs_f = self.forward_model(f.flatten())  # first order Taylor series approximation
        J = np.diag(g_obs_f * (1 - g_obs_f))

        return g_obs_f, J

    def _update_jacobian(self, G_update_rate=1.0):
        g_obs_f, J = self._compute_jacobian()
        if G_update_rate == 1 or not len(
                self.G) or self.G.shape != J.shape:  # either G has not been initialised, or is from different observations
            self.G = J
        else:
            self.G = G_update_rate * J + (1 - G_update_rate) * self.G

    # Log Likelihood Computation -------------------------------------------------------------------------------------

    def lowerbound(self, return_terms=False):
        logp_Df = self._logp_Df()
        logq_f = self._logqf()

        if self.fixed_s:
            logp_s = 0
            logq_s = 0
        else:
            logp_s = self._logps()
            logq_s = self._logqs()

        if self.verbose:
            logging.debug("DLL + logp_f: %.5f, logq_f: %.5f, logp_s-logq_s: %.5f" % (logp_Df, logq_f, logp_s - logq_s))
            # logging.debug("pobs : %.4f, pz: %.4f" % (pobs, pz) )
            # logging.debug("logp_f - logq_f: %.5f. logp_s - logq_s: %.5f" % (logp_f - logq_f, logp_s - logq_s))
            # logging.debug("LB terms without the output scale: %.3f" % (data_ll + logp_f - logq_f))

        lb = logp_Df - logq_f + logp_s - logq_s

        if return_terms:
            return lb, logp_Df, logq_f, logp_s, logq_s

        return lb

    def get_obs_precision(self):
        return self.G.T.dot(np.diag(1.0 / self.Q)).dot(self.G)

    def lowerbound_gradient(self, dim):
        '''
        Gradient of the lower bound on the marginal likelihood with respect to the length-scale of dimension dim.
        '''
        fhat = (self.obs_f - self.mu0)
        invKs_fhat = self.s * self.invK.dot(fhat)

        sigmasq = self.get_obs_precision()

        if self.n_lengthscales == 1 or dim == -1:  # create an array with values for each dimension
            dims = range(self.obs_coords.shape[1])
        else:  # do it for only the dimension dim
            dims = [dim]

        invKs_C = self.s * self.invK.dot(self.obs_C)
        invKs_C_sigmasq = invKs_C.T.dot(sigmasq)

        firstterm = np.zeros(len(dims))
        secondterm = np.zeros(len(dims))
        for d, dim in enumerate(dims):
            kernel_derfactor = self.kernel_derfactor(self.obs_coords[:, dim:dim + 1], self.obs_coords[:, dim:dim + 1],
                                                     self.ls[dim], operator=self.kernel_combination) / self.s
            if self.kernel_combination == '*':
                dKdls = self.K * kernel_derfactor
            else:
                dKdls = kernel_derfactor
            firstterm[d] = invKs_fhat.T.dot(dKdls).dot(invKs_fhat)
            secondterm[d] = np.trace(invKs_C_sigmasq.dot(dKdls))

        if self.n_lengthscales == 1:
            # sum the partial derivatives over all the dimensions
            firstterm = [np.sum(firstterm)]
            secondterm = [np.sum(secondterm)]

        gradient = 0.5 * (firstterm - secondterm)
        return np.array(gradient)

    def ln_modelprior(self):
        # Gamma distribution over each value. Set the parameters of the gammas.
        lnp_gp = - gammaln(self.shape_ls) + self.shape_ls * np.log(self.rate_ls) \
                 + (self.shape_ls - 1) * np.log(self.ls) - self.ls * self.rate_ls
        return np.sum(lnp_gp)

    def _logpt(self):
        # this produces an upper bound on the expected log (a concave function), according to jensen's inequality.
        # Having an upper bound on the variational lower bound could introduce noise when optimising...
        # rho, notrho = self._post_rough(self.obs_f, self.obs_C)
        # logrho = np.log(rho)
        # lognotrho = np.log(notrho)

        logrho, lognotrho, _ = self._post_sample(self.obs_f, np.diag(self.obs_C)[:, None], expectedlog=True)

        return logrho, lognotrho

    def _logpf(self):
        # Note that some terms are cancelled with those in data_ll to simplify
        _, logdet_K = np.linalg.slogdet(self.K)
        D = len(self.obs_f)
        logdet_Ks = -D * self.Elns + logdet_K

        # term below simplifies
        invK_expecF = np.trace(self.invK.dot(self.obs_C) * self.s)
        # invK_expecF = D

        m_invK_m = (self.obs_f - self.mu0).T.dot(self.invK*self.s).dot(self.obs_f-self.mu0)

        return 0.5 * (- np.log(2 * np.pi) * D - logdet_Ks - invK_expecF - m_invK_m)

    def _logp_Df(self):
        """
        Expected joint log likelihood of the data, D, and the latent function, f

        :return:
        """
        # sigma = self.obs_variance()
        #
        # logrho, lognotrho, _ = self._post_sample(self.obs_f, sigma, True)

        # We avoid the sampling step by using the Gaussian approximation to the likelihood to separate the term
        # relating to uncertainty in f from the likelihood function given f, then replacing the Gaussian part with
        # the correct likelihood. This leaves one remaining term, - 0.5 * np.trace(self.u_Lambda.dot(self.uS)), which
        # simplifies when combined with np.trace(self.invKs_mm_uS) in log p(f) to D because inv(uS) = self.invKs_mm + self.u_Lambda
        logrho, lognotrho = self._logpt()

        logdll = self.data_ll(logrho, lognotrho)

        logpf = self._logpf()
        return logpf + logdll

    def data_ll(self, logrho, lognotrho):
        bc = binom(self.obs_total_counts, self.z * self.obs_total_counts)
        logbc = np.log(bc)
        lpobs = np.sum(self.z * self.obs_total_counts * logrho + self.obs_total_counts * (1 - self.z) * lognotrho)
        lpobs += np.sum(logbc)

        data_ll = lpobs
        return data_ll

    def _logqf(self):
        # We want to do this, but we can simplify it, since the x and mean values cancel:
        _, logdet_C = np.linalg.slogdet(self.obs_C)
        D = len(self.obs_f)
        logqf = 0.5 * (- np.log(2 * np.pi) * D - logdet_C - D)
        return logqf

    def _logps(self):
        logprob_s = - gammaln(self.shape_s0) + self.shape_s0 * np.log(self.rate_s0) + (self.shape_s0 - 1) * self.Elns \
                    - self.rate_s0 * self.s
        return logprob_s

    def _logqs(self):
        lnq_s = - gammaln(self.shape_s) + self.shape_s * np.log(self.rate_s) + (self.shape_s - 1) * self.Elns - \
                self.rate_s * self.s

        return lnq_s

    def neg_marginal_likelihood(self, hyperparams, dimension, use_MAP=False):
        '''
        Weight the marginal log data likelihood by the hyper-prior. Unnormalised posterior over the hyper-parameters.

        '''
        if np.any(np.isnan(hyperparams)):
            return np.inf
        if dimension == -1 or self.n_lengthscales == 1:
            self.ls[:] = np.exp(hyperparams)
        else:
            self.ls[dimension] = np.exp(hyperparams)
        if np.any(np.isinf(self.ls)):
            return np.inf

        if np.any(self.ls < 1e-100 * self.initialguess):
            # avoid very small length scales
            return np.inf

        self.reset_kernel()  # regenerate kernel with new length-scales
        self.vb_iter = 0
        # make sure we start again
        # Sets the value of parameters back to the initial guess
        self._init_params(None, True, None)
        self.fit(process_obs=False, optimize=False)
        if self.verbose:
            logging.debug("Inverse output scale: %f" % self.s)

        marginal_log_likelihood = self.lowerbound()

        if use_MAP:
            log_model_prior = self.ln_modelprior()
            lml = marginal_log_likelihood + log_model_prior
        else:
            lml = marginal_log_likelihood
        logging.debug("LML: %f, with Length-scales from %f to %f, after %i iterations" % (
        lml, np.min(self.ls), np.max(self.ls), self.vb_iter))
        return -lml

    def nml_jacobian(self, hyperparams, dimension, use_MAP=False):
        # for the case where this method is called before the model is fitted to the given lengthscale
        if dimension == -1 or self.n_lengthscales == 1:
            if np.any(np.abs(self.ls - np.exp(hyperparams)) > 1e-4):
                self.ls[:] = np.exp(hyperparams)
                self.neg_marginal_likelihood(hyperparams, dimension, use_MAP)
        elif np.any(np.abs(self.ls[dimension] - np.exp(hyperparams)) > 1e-4):
            # don't update if the change is too small
            self.ls[dimension] = np.exp(hyperparams)
            self.neg_marginal_likelihood(hyperparams, dimension, use_MAP)

        gradient = self.lowerbound_gradient(dimension)
        gradient[np.isnan(gradient)] = 1e-20  # remove any values that are likely to cause errors. Positive gradient to
        # encourage the lengthscales causing the problems to be increased, since they are most likely close to zero
        if use_MAP:
            logging.error("MAP not implemented yet with gradient descent methods -- will ignore the model prior.")

        logging.debug('Jacobian of LML: ' + str(np.round(gradient, 4)))
        # logging.debug('Jacobian of LML: largest gradient = %.3f, smallest gradient = %.3f' % (np.max(gradient),
        #                                                                                       np.min(gradient)))

        return -np.array(gradient, order='F')

    # Training methods ------------------------------------------------------------------------------------------------

    def set_training_data(self, obs_coords=None, obs_values=None, totals=None, mu0=None, K=None,
            maxfun=20, use_MAP=False, nrestarts=1, features=None, init_Q_only=False):
        """

        Initialise the model using the current training data, but do not run the training algorithm. Useful to
        initialise observation noise when the covariance function and prior mean may change between iterations when
        the GP is used as part of a larger graphical model. In such cases, call this function with the priors, then
        use subsequent calls to fit to train on new prior means and covariance matrices.

        :param obs_coords: coordinates of observations as an N x D array, where N is number of observations,
        D is number of dimensions
        :param obs_values:
        :param totals:
        :param process_obs:
        :param mu0:
        :param K:
        :param optimize:
        :param maxfun:
        :param use_MAP:
        :param nrestarts:
        :param features:
        :return:
        """
        if features is not None:  # keep the old item features if we pass in none
            self.features = features

        # Initialise the objects that store the observation data
        self._process_observations(obs_coords, obs_values, totals)

        self._init_params(mu0, init_Q_only is False, K)
        self.vb_iter = 0 # don't reset if we don't have new data

        if not len(self.obs_coords):
            return

        if self.verbose:
            logging.debug("GP Classifier VB: prepared for training with max length-scale %.3f and smallest %.3f" % (np.max(self.ls),
                                                                                                       np.min(self.ls)))

    def fit(self, obs_coords=None, obs_values=None, totals=None, process_obs=True, mu0=None, K=None, optimize=False,
            maxfun=20, use_MAP=False, nrestarts=1, features=None, use_median_ls=False):
        '''
        obs_coords -- coordinates of observations as an N x D array, where N is number of observations,
        D is number of dimensions

        TODO: simplify interface by removing process_obs so that if the data is passed in, it is
        always processed.
        '''
        if features is not None: # keep the old item features if we pass in none
            self.features = features

        if optimize:
            return self._optimize(obs_coords, obs_values, totals, process_obs, mu0, K, maxfun, use_MAP, nrestarts,
                                  use_median_ls)

        # Initialise the objects that store the observation data
        if process_obs:
            prev_n_locs = self.n_locs # how many training data points did we have before?
            self._process_observations(obs_coords, obs_values, totals)

            # do we have new training locations?
            new_locations = (features is not None) or (self.n_locs != prev_n_locs)

            if use_median_ls and new_locations:
                self.ls = compute_median_lengthscales(self.obs_coords)

            self._init_params(mu0, new_locations, K)
            self.vb_iter = 0 # reset if we have processed new observations

        elif mu0 is not None or K is not None:  # updated mean but not updated observations
            self._init_params(mu0, False, K)  # don't reset the parameters, but make sure mu0 is updated

        if not process_obs:
            self.max_iter_VB = self.vb_iter + self.max_iter_VB_per_fit

        if not len(self.obs_coords):
            return

        if self.n_obs == 0:
            return

        if self.verbose:
            logging.debug("GP Classifier VB: training with max length-scale %.3f and smallest %.3f" % (np.max(self.ls),
                                                                                                       np.min(self.ls)))
        converged_count = 0
        prev_val = -np.inf
        while converged_count < self.n_converged and self.vb_iter < self.max_iter_VB:
            self._expec_f()

            # update the output scale parameter (also called latent function scale/sigmoid steepness)
            if not self.fixed_s:
                self._expec_s()

            converged, prev_val = self._check_convergence(prev_val)
            converged_count += converged
            if not converged and np.mod(self.vb_iter, self.conv_check_freq) == 0 and converged_count > 0:  # reset the convergence count as the difference has increased again
                converged_count = 0

            if self.verbose:
                logging.debug('Converged iteration count = %i' % converged_count)

            self.vb_iter += 1

        #self.vb_iter -= 1 # the next line repeats the last iteration
        self._update_f()  # this is needed so that L and A match s
        #self.vb_iter += 1

        if self.verbose:
            logging.debug("GP fit complete. Inverse output scale=%.5f" % self.s)

    def _optimize(self, obs_coords, obs_values, totals=None, process_obs=True, mu0=None, K=None, maxfun=25,
                  use_MAP=False, nrestarts=1, use_median_ls=True):

        if process_obs:
            self._process_observations(obs_coords, obs_values, totals)  # process the data here so we don't repeat each call

            if use_median_ls:
                self.ls = compute_median_lengthscales(self.obs_coords)

            self.vb_iter = 0 # reset if we have processed new observations
            self._init_params(mu0, True, K)
            max_iter = self.max_iter_VB_per_fit
            self.max_iter_VB_per_fit = 1
            self.fit(process_obs=False, optimize=False, use_median_ls=False)
            self.max_iter_VB_per_fit = max_iter

        nfits = 0
        min_nlml = np.inf
        best_opt_hyperparams = None
        best_iter = -1
        logging.debug("Optimising length-scale for all dimensions")

        for r in range(nrestarts):
            self.initialguess = np.log(self.ls)
            if self.n_lengthscales == 1:
                self.initialguess = self.initialguess[0]
            logging.debug("Initial length-scale guess in restart %i: %s" % (r, self.ls))

            # res = minimize(self.neg_marginal_likelihood, self.initialguess,
            #                args=(-1, use_MAP,), method='Nelder-Mead',
            #                options={'maxiter':maxfun, 'fatol':1,
            #                         'xatol':1e6, 'disp':True})

            res = minimize(self.neg_marginal_likelihood, self.initialguess,
                           args=(-1, use_MAP,), jac=self.nml_jacobian, method='L-BFGS-B',
                           bounds=Bounds(np.log(1e-3*self.ls), np.log(1e3*self.ls)),
                           options={'maxfun':maxfun, 'maxiter':maxfun, 'gtol':10**(-self.ninput_features), 'disp':True})

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

        if best_iter < r:
            # need to go back to the best result
            self.neg_marginal_likelihood(best_opt_hyperparams, -1, use_MAP=False)

        logging.debug("Optimal value = %.5f with hyper-parameters: %s; found using %i objective fun evals" %
                      (-nlml, self.ls, nfits))
        return self.ls, -min_nlml  # return the log marginal likelihood

    def _expec_f(self):
        '''
        Compute the expected value of f given current q() distributions for other parameters. Could plug in a different
        GP implementation here.
        '''
        diff_G = 0
        G_update_rate = 1.0  # start with full size updates
        # Iterate a few times to get G to stabilise
        for G_iter in range(self.max_iter_G):
            oldG = self.G
            self._update_jacobian(G_update_rate)
            self._update_f()
            prev_diff_G = diff_G  # save last iteration's difference
            if np.isscalar(oldG) or self.G.shape == oldG.shape:
                diff_G = np.max(np.abs(oldG - self.G))
            else:
                diff_G = np.inf # if the shape has changed, we have changed the observations we are fitting to
            # Use a smaller update size if we get stuck oscillating about the solution
            if np.abs(np.abs(diff_G) - np.abs(prev_diff_G)) < 1e-3 and G_update_rate > 0.1:
                G_update_rate *= 0.9
            if self.verbose:
                logging.debug("Iterating over G: diff was %.5f in G-iteration %i; update rate = %f" % (diff_G, G_iter, G_update_rate))
            if diff_G < self.conv_threshold_G and G_iter > 0:
                break;
        if G_iter >= self.max_iter_G - 1:
            if self.verbose:
                logging.debug("G did not converge: diff was %.5f" % diff_G)

    def _update_f(self):
        self.KsG = self.Ks.dot(self.G.T, out=self.KsG)
        self.Cov = self.KsG.T.dot(self.G.T, out=self.Cov)
        self.Cov[range(self.Cov.shape[0]), range(self.Cov.shape[0])] += self.Q

        # use the estimate given by the Taylor series expansion
        z0 = self.forward_model(self.obs_f) + self.G.dot(self.mu0 - self.obs_f)

        self.L = cholesky(self.Cov, lower=True, check_finite=False, overwrite_a=True)
        B = solve_triangular(self.L, (self.z - z0), lower=True, overwrite_b=True, check_finite=False)
        self.A = solve_triangular(self.L, B, lower=True, trans=True, overwrite_b=False, check_finite=False)
        self.obs_f = self.KsG.dot(self.A, out=self.obs_f) + self.mu0  # need to add the prior mean here?
        V = solve_triangular(self.L, self.KsG.T, lower=True, overwrite_b=True, check_finite=False)
        self.obs_C = self.Ks - V.T.dot(V, out=self.obs_C)

    def _expec_s(self):
        self.old_s = self.s
        invK_expecFF = self.invK.dot(self.obs_C + self.obs_f.dot(self.obs_f.T) \
                                 - self.mu0.dot(self.obs_f.T) - self.obs_f.dot(self.mu0.T) + self.mu0.dot(self.mu0.T))
        if not self.fixed_s:
            self.rate_s = self.rate_s0 + 0.5 * np.trace(invK_expecFF)
        # Update expectation of s. See approximations for Binary Gaussian Process Classification, Hannes Nickisch
        self.s = self.shape_s / self.rate_s
        self.Elns = psi(self.shape_s) - np.log(self.rate_s)
        if self.verbose:
            logging.debug("Updated inverse output scale: " + str(self.s))
        self.Ks = self.K / self.s

    def _check_convergence(self, prev_val):
        if self.uselowerbound and np.mod(self.vb_iter, self.conv_check_freq) == 0:
            oldL = prev_val
            L = self.lowerbound()
            converged = check_convergence(L, oldL, self.conv_threshold, True, self.vb_iter,
                                          self.verbose, 'GP Classifier VB lower bound')
            current_value = L
        elif not self.uselowerbound and np.mod(self.vb_iter, self.conv_check_freq) == 0:
            diff = np.max(np.abs(self.obs_f - prev_val))
            if self.verbose:
                logging.debug('GP Classifier VB obs_f diff = %f at iteration %i' % (diff, self.vb_iter))

            sdiff = np.abs(self.old_s - self.s) / self.s
            if self.verbose:
                logging.debug('GP Classifier VB s diff = %.5f' % sdiff)

            diff = np.max([diff, sdiff])
            current_value = self.obs_f
            converged = (diff < self.conv_threshold) & (self.vb_iter > 2)
        else:
            return False, prev_val  # not checking in this iteration, return the old value and don't converge

        return (converged & (self.vb_iter + 1 >= self.min_iter_VB)), current_value

    # Prediction methods ---------------------------------------------------------------------------------------------

    def predict(self, out_feats=None, out_idxs=None, K_star=None, K_starstar=None, variance_method='rough',
                expectedlog=False, mu0_output=None, reuse_output_kernel=False):
        '''
        Evaluate the function posterior mean and variance at the given co-ordinates using the 2D squared exponential
        kernel

        Parameters
        ----------

        reuse_output_kernel : can be switched on to reuse the output kernel to save computational cost when the
        out_feats object is the same over many calls to predict(), and the lengthscale and covariance function
        do not change between calls.

        '''
        self.predict_f(out_feats, out_idxs, K_star, K_starstar, mu0_output, reuse_output_kernel)

        if variance_method == 'sample' or expectedlog:
            if variance_method == 'rough':
                logging.warning(
                    "Switched to using sample method as expected log requested. No quick method is available.")

            # Approximate the expected value of the variable transformed through the sigmoid.
            m_post, not_m_post, v_post = self._post_sample(self.f, self.v, expectedlog)
        elif variance_method == 'rough' and not expectedlog:
            m_post, _ = self._post_rough(self.f, self.v)

        if expectedlog:
            if variance_method == 'sample':
                return m_post, not_m_post, v_post
            else:
                return m_post, not_m_post
        elif variance_method == 'sample':
            return m_post, v_post
        else:
            return m_post

    def predict_grid(self, nx, ny, variance_method='rough', mu0_output=None):
        nout = nx * ny
        outputx = np.tile(np.arange(nx, dtype=np.float).reshape(nx, 1), (1, ny)).reshape(nout)
        outputy = np.tile(np.arange(ny, dtype=np.float).reshape(1, ny), (nx, 1)).reshape(nout)
        return self.predict(out_feats=[outputx, outputy], variance_method=variance_method, mu0_output=mu0_output)

    def _get_training_cov(self,):
        # return the covariance matrix for training points to inducing points (if used) and the variance of the training points.
        return self.K, self.K

    def _get_training_feats(self):
        return self.obs_coords

    def predict_f(self, out_feats=None, out_idxs=None, K_star=None, K_starstar=None, mu0_output=None,
                  reuse_output_kernel=False, full_cov=False):
        # Establish the output covariance matrices
        if K_star is not None and K_starstar is not None:
            # use the matrices passed in directly
            self.K_star = K_star
            self.K_starstar = K_starstar

        elif out_feats is not None and K_star is None and K_starstar is None:
            # should only change if lengthscale self.ls or cov_type change
            if not reuse_output_kernel or self.out_feats is None or np.any(out_feats != self.out_feats) or self.K_star is None:
                if out_feats.shape[0] == self.ninput_features and out_feats.shape[1] != self.ninput_features:
                    out_feats = out_feats.T
                self.out_feats = out_feats
                # compute kernels given the feature vectors supplied
                out_feats_arr = np.array(out_feats).astype(float)
                self.K_star = self.kernel_func(out_feats_arr, self.ls, self._get_training_feats(),
                                               operator=self.kernel_combination, n_threads=self.n_threads)
                if full_cov:
                    self.K_starstar = self.kernel_func(out_feats_arr, self.ls, out_feats_arr,
                                                       operator=self.kernel_combination, n_threads=self.n_threads)
                else:
                    self.K_starstar = 1.0  # assuming that the kernel function places ones along diagonals
            else:
                pass  # we reuse the previous self.K_star and self.K_starstar values

        elif out_feats is None and K_star is None and K_starstar is None:
            # use the training feature vectors
            self.K_star, self.K_starstar = self._get_training_cov()
            if not full_cov:
                self.K_starstar = np.diag(self.K_starstar)

        else:
            # other combinations are invalid
            logging.error(
                'Invalid combination of parameters for predict(): please supply either out_feats OR (K_star AND K_starstar)')
            return

        K_star = self.K_star
        K_starstar = self.K_starstar

        # Select the data points from the complete set using the indexes
        if out_idxs is not None:
            K_star = K_star[out_idxs, :]
            if not np.isscalar(K_starstar):
                if full_cov:
                    K_starstar = K_starstar[out_idxs, :][:, out_idxs]
                else:
                    K_starstar = K_starstar[out_idxs]

        noutputs = K_star.shape[0]

        # The output mean
        if mu0_output is None:
            self.mu0_output = np.zeros((noutputs, 1)) + self.mu0_default
        else:
            self.mu0_output = np.reshape(mu0_output, (mu0_output.shape[0], 1))

        # predict f for the given kernels and mean
        if self.verbose:
            logging.debug("GPClassifierVB predicting f")

        Ks_starstar = K_starstar / self.s
        Ks_star = K_star / self.s
        self.f, self.v = self._expec_f_output(Ks_starstar, Ks_star, self.mu0_output, full_cov, reuse_output_kernel)

        if full_cov:
            v = np.diag(self.v) # self.v is full covariance. Get the diags to check it is okay.
        else:
            v = self.v
        if np.any(v < 0):
            logging.error("Negative variance in GPClassifierVB._predict_f(), %f" % np.min(v))

        return self.f, self.v

    def _expec_f_output(self, Ks_starstar, Ks_star, mu0, full_cov=False, reuse_output_kernel=False):
        """
        Compute the expected value of f and the variance or covariance of f
        :param Ks_starstar: prior variance at the output points (scalar or 1-D vector), or covariance if full_cov==True.
        :param Ks_star: covariance between output points and training points
        :param mu0: prior mean for output points
        :param full_cov: set to True to compute the full posterior covariance between output points
        :return f, C: posterior expectation of f, variance or covariance of the output locations.
        """
        f = Ks_star.dot(self.G.T).dot(self.A) + mu0
        V = solve_triangular(self.L, self.G.dot(Ks_star.T), lower=True, overwrite_b=True, check_finite=False)

        if full_cov:
            C = Ks_starstar - V.T.dot(V)
        else:
            C = (Ks_starstar - np.sum(V ** 2, axis=0))[:, None]
        return f, C

    def _post_rough(self, f, C=None):
        """

        :param f:
        :param C: Can be a full covariance or a vector of variance
        :return:
        """

        if C is None:
            v = 0
        elif C.ndim == 2:
            v = np.diag(C)
        else:
            v = C

        k = 1.0 / np.sqrt(1 + (np.pi * v / 8.0))
        m_post = sigmoid(k * f)
        not_m_post = sigmoid(-k * f)

        return m_post, not_m_post

    def _post_sample(self, f, v, expectedlog):
        # draw samples from a Gaussian with mean f and variance v
        f_samples = np.random.normal(loc=f, scale=np.sqrt(v), size=(len(f), 500))
        rho_samples = self.forward_model(f_samples)
        rho_samples = temper_extreme_probs(rho_samples)
        rho_not_samples = 1 - rho_samples
        if expectedlog:
            rho_samples = np.log(rho_samples)
            rho_not_samples = np.log(rho_not_samples)
        m_post = np.mean(rho_samples, axis=1)[:, np.newaxis]
        not_m_post = np.mean(rho_not_samples, axis=1)[:, np.newaxis]
        v_post = np.mean((rho_samples - m_post) ** 2, axis=1)[:, np.newaxis]

        return m_post, not_m_post, v_post


if __name__ == '__main__':
    logging.warning('Caution: this is not a proper test of the whole algorithm.')
    from scipy.stats import multivariate_normal as mvn

    # run some tests on the learning algorithm

    N = 100

    # generate test ground truth
    s = 10
    mean = np.zeros(N)

    gridsize = 1000.0
    ls = 10.0

    x_all = np.arange(N) / float(N) * gridsize
    y_all = np.arange(N) / float(N) * gridsize

    ddx = x_all[:, np.newaxis] - x_all[np.newaxis, :]
    ddy = y_all[:, np.newaxis] - y_all[np.newaxis, :]

    Kx = np.exp(-ddx ** 2 / ls)
    Ky = np.exp(-ddy ** 2 / ls)
    K = Kx * Ky

    K += np.eye(N) * 1e-6

    K_over_s = K / s
    invK = np.linalg.inv(K)
    L = cholesky(K, lower=True, check_finite=False)

    nsamples = 500
    shape0 = 1.0
    rate0 = 1.0

    # now try to infer the output scale given the ground truth
    shape = shape0
    rate = rate0
    for i in range(nsamples):
        f_true = mvn.rvs(mean=mean, cov=K_over_s)[:, np.newaxis]

        shape += 0.5 * len(f_true)
        rate += 0.5 * np.trace(solve_triangular(L, solve_triangular(L, f_true.dot(f_true.T),
                                                                    lower=True, overwrite_b=True, check_finite=False),
                                                trans=True, overwrite_b=True, check_finite=False))
    post_s = shape / rate
    print(shape)
    print(rate)
    print("Posterior estimate of s is %f" % post_s)
