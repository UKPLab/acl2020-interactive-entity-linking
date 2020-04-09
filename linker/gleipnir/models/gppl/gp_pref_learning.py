'''
Created on 18 May 2016

@author: simpson
'''
import numpy as np
from scipy.stats import norm, multivariate_normal as mvn
from scipy.sparse import coo_matrix, issparse, hstack
import logging
from gleipnir.models.gppl.gp_classifier_vb import coord_arr_to_1d, coord_arr_from_1d, temper_extreme_probs
from gleipnir.models.gppl.gp_classifier_svi import GPClassifierSVI

def get_unique_locations(obs_coords_0, obs_coords_1, mu_0=None, mu_1=None):
    if issparse(obs_coords_0) or issparse(obs_coords_1):
        uidxs_0 = []
        pref_vu = []

        for r, row in enumerate(obs_coords_0):
            print("%i out of %i" % (r, obs_coords_0.shape[0]))
            idx = row == obs_coords_0[uidxs_0]
            if not np.sum(idx):
                uidxs_0.append(r)
                pref_vu.append(len(uidxs_0) - 1)
            else:
                pref_vu.append(np.argwhere(idx)[0])

        len_0 = obs_coords_0.shape[0]
        uidxs_1 = []
        for r, row in enumerate(obs_coords_1):
            print("%i out of %i" % (r, obs_coords_0.shape[0]))
            idx = row == obs_coords_0[uidxs_0]
            if not np.sum(idx):
                idx = row == obs_coords_1[uidxs_1]
                if not np.sum(idx):
                    uidxs_1.append(r + len_0)
                    pref_vu.append(len(uidxs_1) - 1)
                else:
                    pref_vu.append(np.argwhere(idx)[0] + len_0)
            else:
                pref_vu.append(np.argwhere(idx)[0])

        # convert urows to a sparse matrix
        obs_coords = hstack((obs_coords_0[uidxs_0], obs_coords_1[uidxs_1]), format='csc')
        uidxs = np.concatenate((uidxs_0, np.array(uidxs_1) + len_0))
    else:
        coord_rows_0 = coord_arr_to_1d(obs_coords_0)
        coord_rows_1 = coord_arr_to_1d(obs_coords_1)
        all_coord_rows = np.concatenate((coord_rows_0, coord_rows_1), axis=0)
        _, uidxs, pref_vu = np.unique(all_coord_rows, return_index=True, return_inverse=True) # get unique locations

        # Record the coordinates of all points that were compared
        obs_coords = np.concatenate((obs_coords_0, obs_coords_1), axis=0)[uidxs]

    # Record the indexes into the list of coordinates for the pairs that were compared
    pref_v = pref_vu[:obs_coords_0.shape[0]]
    pref_u = pref_vu[obs_coords_0.shape[0]:]

    if mu_0 is not None and mu_1 is not None:
        mu_vu = np.concatenate((mu_0, mu_1), axis=0)[uidxs]
        return obs_coords, pref_v, pref_u, mu_vu
    else:
        return obs_coords, pref_v, pref_u

def pref_likelihood(fmean, fvar=None, subset_idxs=[], v=[], u=[], return_g_f=False):
    '''
    f - should be of shape nobs x 1

    This returns the probability that each pair has a value of 1, which is referred to as Phi(z)
    in the chu/ghahramani paper, and the latent parameter referred to as z in the chu/ghahramani paper.
    In this work, we use z to refer to the observations, i.e. the fraction of comparisons of a given pair with
    value 1, so use a different label here.
    '''
    if len(subset_idxs):
        if len(v) and len(u):
            # keep only the pairs that reference two items in the subet
            # pair_subset = np.in1d(v, subset_idxs) & np.in1d(u, subset_idxs)
            v = v[subset_idxs]  #pair_subset]
            u = u[subset_idxs]  #pair_subset]
        else:
            fmean = fmean[subset_idxs]

    if fmean.ndim < 2:
        fmean = fmean[:, np.newaxis]

    if fvar is None:
        fvar = 2.0
    else:
        if fvar.ndim < 2:
            fvar = fvar[:, np.newaxis]        
        fvar = fvar + 2.0

    if np.any(fvar < 0):
        logging.warning('There was a negative variance in the pref likelihood! %s' % fvar)
        fvar[fvar < 0] = 0

    if len(v) and len(u):
        g_f = (fmean[v, :] - fmean[u, :]) / np.sqrt(fvar)
    else: # provide the complete set of pairs
        g_f = (fmean - fmean.T) / np.sqrt(fvar)

    phi = norm.cdf(g_f) # the probability of the actual observation, which takes g_f as a parameter. In the
    # With the standard GP density classifier, we can skip this step because
    # g_f is already a probability and Phi(z) is a Bernoulli distribution.
    if return_g_f:
        return phi, g_f
    else:
        return phi

class GPPrefLearning(GPClassifierSVI):
    '''
    Preference learning with GP, with variational inference implementation. Can use stochastic variational inference.

    Redefines:
    - Calculations of the Jacobian, referred to as self.G
    - Nonlinear forward model, "sigmoid"
    - Process_observations:
     - Observations, self.z. Observations now consist not of a count at a point, but two points and a label.
     - self.obsx and self.obsy refer to all the locations mentioned in the observations.
     - self.Q is the observation covariance at the observed locations, i.e. the
    - Lower bound?
    '''

    pref_v = [] # the first items in each pair -- index to the observation coordinates in self.obsx and self.obsy
    pref_u = [] # the second items in each pair -- indices to the observations in self.obsx and self.obsy

    def __init__(self, ninput_features, mu0=0, shape_s0=2, rate_s0=2, shape_ls=10, rate_ls=0.1, ls_initial=None,
        kernel_func='matern_3_2', kernel_combination='*',
        max_update_size=1000, ninducing=500, use_svi=True, delay=10, forgetting_rate=0.7, verbose=False, fixed_s=False):

        # We set the function scale and noise scale to the same value so that we assume apriori that the differences
        # in preferences can be explained by noise in the preference pairs or the latent function. Ordering patterns
        # will change this balance in the posterior.

        #self.sigma = 1 # controls the observation noise. Equivalent to the output scale of f? I.e. doesn't it have the
        # same effect by controlling the amount of noise that is permissible at each point? If so, I think we can fix this
        # to 1.
        # By approximating the likelihood distribution with a Gaussian, the covariance of the approximation is the
        # inverse Hessian of the negative log likelihood. Through moment matching self.Q with the likelihood covariance,
        # we can compute sigma?

        if shape_s0 <= 0:
            shape_s0 = 0.5
        if rate_s0 <= 0:
            rate_s0 = 0.5

        super(GPPrefLearning, self).__init__(ninput_features, mu0, shape_s0, rate_s0, shape_ls, rate_ls, ls_initial,
         kernel_func, kernel_combination,
         max_update_size, ninducing, use_svi, delay, forgetting_rate, verbose=verbose, fixed_s=fixed_s)

    # Initialisation --------------------------------------------------------------------------------------------------

    def _init_prior_mean_f(self, z0):
        self.mu0_default = z0 # for preference learning, we pass in the latent mean directly

    def _init_obs_prior(self):
        # to make a and b smaller and put more weight onto the observations, increase v_prior by increasing rate_s0/shape_s0

        if self.mu0 is None:
            mu0 = self.mu0_default
        else:
            mu0 = self.mu0

        # OLD VERSION:
        # I think that we should really sample using the full covariance here, since some values are bound to be close
        # together given the prior covariance. However, in practice this is much slower to sample, so we use a diagonal
        # as an approximation, which may  under-estimate variance, since the resulting Q is a combination of
        # a term with too low observation variance + too high model variance. Hence we exaggerated the amount learned
        # from similar points, reducing the effect of the prior covariance.
        # f_prior_var = self.rate_s0/self.shape_s0
        #m_prior, not_m_prior, v_prior = self._post_sample(mu0, f_prior_var, False, None, self.pref_v, self.pref_u)
        # When we used this version we were coincidentally compensating by adding the variance on mistakenly...

        if not len(self.pref_v):
            self.nu0 = []
            return

        # NEW VERSION:
        m_prior, _, v_prior = self._post_sample(mu0, self.K_mm/self.s, False, self.K_nm, self.pref_v, self.pref_u)
        if not np.any(np.nonzero(self.mu0)):
            m_prior = 0.5

        # find the beta parameters
        a_plus_b = 1.0 / (v_prior / (m_prior*(1-m_prior))) - 1
        a = (a_plus_b * m_prior)
        b = (a_plus_b * (1-m_prior))

        self.nu0 = np.array([b, a])
        #if self.verbose:
        #    logging.debug("Prior parameters for the observed pairwise preference variance are: %s" % str(self.nu0))

    def _init_obs_f(self):
        # Mean probability at observed points given local observations
        if self.obs_f is None:  # don't reset if we are only adding more data
            self.obs_f = np.zeros((self.n_locs, 1)) + self.mu0
        elif self.obs_f.shape[0] < self.n_locs:
            prev_obs_f = self.obs_f
            self.obs_f = np.zeros((self.n_locs, 1)) + self.mu0
            self.obs_f[:prev_obs_f.shape[0], :] = prev_obs_f


    def _init_obs_mu0(self, mu0):
        if mu0 is None or not len(mu0):
            self.mu0 = np.zeros((self.n_locs, 1)) + self.mu0_default
        else:
            self.mu0 = np.array(mu0)
            if self.mu0.ndim == 1:
                self.mu0 = self.mu0[:, None]

            if len(self.pref_v):
                self.mu0_1 = self.mu0[self.pref_v, :]
                self.mu0_2 = self.mu0[self.pref_u, :]

        self.Ntrain = self.pref_u.size

    # Input data handling ---------------------------------------------------------------------------------------------

    def _count_observations(self, obs_coords, _, poscounts, totals):
        '''
        obs_coords - a tuple with two elements, the first containing the list of coordinates for the first items in each
        pair, and the second containing the coordinates of the second item in the pair.
        '''
        obs_coords_0 = np.array(obs_coords[0])
        obs_coords_1 = np.array(obs_coords[1])
        if obs_coords_0.ndim == 1:
            obs_coords_0 = obs_coords_0[:, np.newaxis]
        if obs_coords_1.ndim == 1:
            obs_coords_1 = obs_coords_1[:, np.newaxis]

        # duplicate locations should be merged and the number of duplicates counted
        #poscounts = poscounts.astype(int)
        totals = totals.astype(int)

        if self.features is not None:
            self.obs_uidxs = np.arange(self.features.shape[0])
            self.pref_v = obs_coords_0.flatten()
            self.pref_u = obs_coords_1.flatten()
            self.n_obs = len(self.pref_v)
            self.obs_coords = self.features
            return poscounts, totals
        else:
            # TODO: This code could be merged with get_unique_locations()
            ravelled_coords_0 = coord_arr_to_1d(obs_coords_0)# Ravel the coordinates
            ravelled_coords_1 = coord_arr_to_1d(obs_coords_1)

            # get unique keys
            all_ravelled_coords = np.concatenate((ravelled_coords_0, ravelled_coords_1), axis=0)
            uravelled_coords, origidxs, keys = np.unique(all_ravelled_coords, return_index=True, return_inverse=True)

            keys_0 = keys[:len(ravelled_coords_0)]
            keys_1 = keys[len(ravelled_coords_0):]

            # SWAP PAIRS SO THEY ALL HAVE LOWEST COORD FIRST so we can count prefs for duplicate location pairs
            idxs_to_swap = keys_0 < keys_1
            swap_coords_0 = keys_0[idxs_to_swap]
            poscounts[idxs_to_swap] = totals[idxs_to_swap] - poscounts[idxs_to_swap]

            keys_0[idxs_to_swap] = keys_1[idxs_to_swap]
            keys_1[idxs_to_swap] = swap_coords_0

            grid_obs_counts = coo_matrix((totals, (keys_0, keys_1)) ).toarray()
            grid_obs_pos_counts = coo_matrix((poscounts, (keys_0, keys_1)) ).toarray()

            nonzero_v, nonzero_u = grid_obs_counts.nonzero() # coordinate key pairs with duplicate pairs removed

            nonzero_all = np.concatenate((nonzero_v, nonzero_u), axis=0)
            ukeys, pref_vu = np.unique(nonzero_all, return_inverse=True) # get unique locations

            self.obs_uidxs = origidxs[ukeys] # indexes of unique observation locations into the original input data

            # Record the coordinates of all points that were compared
            self.obs_coords = coord_arr_from_1d(uravelled_coords[ukeys], obs_coords_0.dtype,
                                            dims=(len(ukeys), obs_coords_0.shape[1]))

            # Record the indexes into the list of coordinates for the pairs that were compared
            self.pref_v = pref_vu[:len(nonzero_v)]
            self.pref_u = pref_vu[len(nonzero_v):]
            self.n_obs = len(self.pref_v)

            # Return the counts for each of the observed pairs
            pos_counts = grid_obs_pos_counts[nonzero_v, nonzero_u]
            total_counts = grid_obs_counts[nonzero_v, nonzero_u]
            return pos_counts, total_counts

    # Mapping between latent and observation spaces -------------------------------------------------------------------

    def forward_model(self, fmean=None, fvar=None, subset_idxs=[], v=[], u=[], return_g_f=False):
        '''
        f - should be of shape nobs x 1

        This returns the probability that each pair has a value of 1, which is referred to as Phi(z)
        in the chu/ghahramani paper, and the latent parameter referred to as z in the chu/ghahramani paper.
        In this work, we use z to refer to the observations, i.e. the fraction of comparisons of a given pair with
        value 1, so use a different label here.
        '''
        if fmean is None:
            fmean = self.obs_f
        if len(v) == 0:
            v = self.pref_v
        if len(u) == 0:
            u = self.pref_u

        return pref_likelihood(fmean, fvar, subset_idxs, v, u, return_g_f)

    def _compute_jacobian(self, f=None, data_idx_i=None):

        if f is None:
            f = self.obs_f

        phi, g_mean_f = self.forward_model(f, return_g_f=True) # first order Taylor series approximation
        J = 1 / (2*np.pi)**0.5 * np.exp(-g_mean_f**2 / 2.0) * np.sqrt(0.5)

        if data_idx_i is not None and hasattr(self, 'data_obs_idx_i') and len(self.data_obs_idx_i):
            obs_idxs = data_idx_i[None, :]
            J = J[self.data_obs_idx_i, :]
            s = (self.pref_v[self.data_obs_idx_i, np.newaxis]==obs_idxs).astype(int) -\
                                                    (self.pref_u[self.data_obs_idx_i, np.newaxis]==obs_idxs).astype(int)
        else:
            obs_idxs = np.arange(self.n_locs)[None, :]
            s = (self.pref_v[:, np.newaxis]==obs_idxs).astype(int) - (self.pref_u[:, np.newaxis]==obs_idxs).astype(int)

        J = J * s

        return phi, J

    def _update_jacobian(self, G_update_rate=1.0):
        phi, J = self._compute_jacobian(data_idx_i=self.data_idx_i)

        if self.G is None or not np.any(self.G) or self.G.shape != J.shape:
            # either G has not been initialised, or is from different observations:
            self.G = J
        else:
            self.G = G_update_rate * J + (1 - G_update_rate) * self.G

        return phi

    # Training methods ------------------------------------------------------------------------------------------------

    def fit(self, items1_coords=None, items2_coords=None, item_features=None, preferences=None, totals=None,
            process_obs=True, mu0=None, K=None, optimize=False, input_type='binary', use_median_ls=False):
        """
        Train the model given a set of preference pairs.
        :param items1_coords:
        :param items2_coords:
        :param item_features:
        :param preferences: Preferences by default (when input_type='binary' can be 1 = item 1 is preferred to item 2,
        or 0 = item 2 is preferred to item 1, 0.5 = no preference, or values in between.
        For preferences that are not normalised to between 0 and 1, the value of input_type needs to be set.
        If the input type is 'zero-centered', the scores range between -1 and 1, where -1 means that the second item
        is preferred and 1 means that the first item is preferred.
        input_type -- c
        :param totals:
        :param process_obs:
        :param mu0: the prior mean of the latent preference function. The value should be a vector (ndarray) of the
        same size as item_features, where each entry of the vector is the prior mean for the corresponding item in
        item_features.
        :param K:
        :param optimize: set to True to use maximum likelihood 2 optimisation of the length-scales
        :param input_type: can be 'binary' or 'zero-centered'. Binary preferences must be [0,1],
        where 0 indicates that item 2 is preferred and 1 indicates that item 1 is preferred.
        Zero-centered perferences have value 1 to indicate item 1 is preferred, value -1 to indicate item 2 is
        preferred, and 0 to indicate no preference.
        :return:
        """
        pref_values_in_input = np.unique(preferences)
        if process_obs and input_type == 'binary' and np.sum((pref_values_in_input < 0) | (pref_values_in_input > 1)):
            raise ValueError('Binary input preferences specified but the data contained the values %s' % pref_values_in_input)
        elif process_obs and input_type == 'zero-centered' and np.sum((pref_values_in_input < -1) | (pref_values_in_input > 1)):
            raise ValueError('Zero-centered input preferences specified but the data contained the values %s' % pref_values_in_input)
        elif process_obs and input_type == 'zero-centered':
            #convert them to [0,1]
            preferences += 1
            preferences /= 2.0
        elif process_obs and input_type != 'binary':
            raise ValueError('input_type for preference labels must be either "binary" or "zero-centered"')

        #TODO: bug fix: if the same object is reused with different set of items, there is a crash because K_nm is not renewed.

        super(GPPrefLearning, self).fit((items1_coords, items2_coords), preferences, totals, process_obs,
                                mu0=mu0, K=K, optimize=optimize, use_median_ls=use_median_ls, features=item_features)

    def set_training_data(self, items1_coords=None, items2_coords=None, item_features=None, preferences=None, totals=None,
            mu0=None, K=None, input_type='binary', init_Q_only=False):
        '''
        preferences -- Preferences by default can be 1 = item 1 is preferred to item 2,
        or 0 = item 2 is preferred to item 1, 0.5 = no preference, or values in between.
        For preferences that are not normalised to between 0 and 1, the value of input_type needs to be set.
        input_type -- can be 'binary', meaning preferences must be [0,1], or 'zero-centered' meaning that value 1
        indicates item 1 is preferred, value -1 indicates item 2 is preferred, and 0 indicates no preference. The value
        are converted internally to [0,1].
        '''
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

        super(GPPrefLearning, self).set_training_data((items1_coords, items2_coords), preferences, totals,
                                        mu0=mu0, K=K, init_Q_only=init_Q_only, features=item_features)

    def _update_sample_idxs(self):
        super(GPPrefLearning, self)._update_sample_idxs()
        self.data_idx_i = np.unique([self.pref_v[self.data_obs_idx_i], self.pref_u[self.data_obs_idx_i]])

    # Prediction methods ---------------------------------------------------------------------------------------------
    def predict(self, out_feats=None, item_0_idxs=None, item_1_idxs=None, K_star=None, K_starstar=None,
                expectedlog=False, mu0_out=None, reuse_output_kernel=False, return_var=True):
        """
        Predict pairwise labels for pairs of items specified by their indices into a list of feature vectors.
        :param item_0_idxs: A list of item indices for the first items in the pairs.
        :param item_1_idxs: A list of item indices for the second items in the pairs.
        :param out_feats: A list of feature vectors for the items. If set to None, the feature vectors passed in using fit() will be used.
        :param K_star: A pre-computed covariance matrix between the test and training items. If set to None, it will be computed automatically if required.
        :param K_starstar: A pre-computed covariance matrix for the predicted items.
        :param expectedlog: return the expected log probability instead of the marginal probability
        :param mu0_out: prior mean preference function values.
        :param reuse_output_kernel: set to True if out_feats does not change between calls to predict(). This will avoid re-computing the covariance matrix in each call to predict().
        :param return_var: return the second-order variance in the prediction probability
        :return p [, var]: probability that first item in pair is preferred to second item [, variance in the probability]
        """

        # predict f for all the rows in out_feats or K_star if these variables are not None, otherwise error.
        f, C = self.predict_f(out_feats, None, K_star, K_starstar, mu0_out, reuse_output_kernel, full_cov=True)

        m_post, not_m_post = self._post_rough(f, C, item_0_idxs, item_1_idxs)

        if return_var:
            if self.use_svi:
                _, _, v_post = self._post_sample(self.mu0_output, self.uS, K_star=self.K_star, v=item_0_idxs, u=item_1_idxs)
            else:
                _, _, v_post = self._post_sample(f, self.uS, K_star=C, v=item_0_idxs, u=item_1_idxs)

        if expectedlog:
            m_post = np.log(m_post)
            not_m_post = np.log(not_m_post)
            if return_var:
                return m_post, not_m_post, v_post
            else:
                return m_post, not_m_post
        elif return_var:
            return m_post, v_post
        else:
            return m_post


    def predict_pairs_from_features(self, out_feats=None, out_1_feats=None, mu0_out=None, mu0_out_1=None,
                                    expectedlog=False, return_var=True):
        """
        Predict pairwise labels by passing in two lists of feature vectors for the first and second items in the pairs.
        This is different to the usual predict() method, which predicts pairwise labels given the indices of items into
        a set of features.

        :param out_feats: list of feature vectors of the first items in the pairs
        :param out_1_feats: list of feature vectors of the second items in the pairs
        :param mu0_out: prior mean preference function values for the first items in the pairs
        :param mu0_out_1: prior mean preference function values for the second items in the pairs
        :param expectedlog: return the expected log probability instead of the marginal probability
        :param return_var: return the second-order variance in the prediction probability
        :return p [, var]: probability that first item in pair is preferred to second item [, variance in the probability]
        """
        out_feats, item_0_idxs, item_1_idxs, mu0_out = get_unique_locations(out_feats, out_1_feats, mu0_out, mu0_out_1)
        return self.predict(out_feats, item_0_idxs, item_1_idxs, mu0_out)


    def _logpt(self):
        # logrho, lognotrho, _ = self._post_sample(self.obs_f, None, True, self.K_nm, self.pref_v, self.pref_u)

        rho = pref_likelihood(self.obs_f, v=self.pref_v, u=self.pref_u)
        rho = temper_extreme_probs(rho)
        logrho = np.log(rho)
        lognotrho = np.log(1 - rho)

        # if we use the Gaussian approximation rather than the true likelihood, there would be a -0.5*tr(CQ^-1) term
        # to take care of the variance in f. However, I think we have dropped it because computing C is expensive???

        return logrho, lognotrho

    def _post_sample(self, f_mean, f_var=None, expectedlog=False, K_star=None, v=None, u=None):

        if v is None:
            v = self.pref_v
        if u is None:
            u = self.pref_u

        if not len(v):
            # no prefereces passed in for training!
            return [], [], []

        if np.isscalar(f_mean):
            N = 1
        else:
            N = f_mean.shape[0]

        # since we don't have f_cov
        if K_star is not None and self.use_svi:
            #sample the inducing points because we don't have full covariance matrix. In this case, f_cov should be Ks_nm
            f_samples = mvn.rvs(mean=self.um_minus_mu0.flatten(), cov=f_var, size=1000).T
            f_samples = K_star.dot(self.invK_mm).dot(f_samples) + f_mean
        elif K_star is not None:
            f_samples = mvn.rvs(mean=f_mean.flatten(), cov=K_star, size=1000).T
        else:
            f_samples = np.random.normal(loc=f_mean, scale=np.sqrt(f_var), size=(N, 1000))

        # g_f = (f_samples[v, :] - f_samples[u, :])  / np.sqrt(2)
        # phi = norm.cdf(g_f) # the probability of the actual observation, which takes g_f as a parameter. In the

        if N == 1:
            phi = self.forward_model(f_samples, v=[0], u=[0])
        else:
            phi = self.forward_model(f_samples, v=v, u=u)

        phi = temper_extreme_probs(phi)
        if expectedlog:
            phi = np.log(phi)
            notphi = np.log(1-phi)
        else:
            notphi = 1 - phi

        m_post = np.mean(phi, axis=1)[:, np.newaxis]
        not_m_post = np.mean(notphi, axis=1)[:, np.newaxis]
        v_post = np.var(phi, axis=1)[:, np.newaxis]
        v_post = temper_extreme_probs(v_post, zero_only=True)
        v_post[m_post * (1 - not_m_post) <= 1e-7] = 1e-8 # fixes extreme values to sensible values. Don't think this is needed and can lower variance?

        return m_post, not_m_post, v_post

    def _post_rough(self, f_mean, f_cov=None, pref_v=None, pref_u=None):
        '''
        When making predictions, we want to predict the probability of each listed preference pair.
        Use a solution given by applying the forward model to the mean of the latent function --
        ignore the uncertainty in f itself, considering only the uncertainty due to the noise sigma.
        '''
        if pref_v is None:
            pref_v = self.pref_v
        if pref_u is None:
            pref_u = self.pref_u

        # to remedy this. However, previously the maths seemed to show it wasn't needed at all?
        if f_cov is None:
            m_post = self.forward_model(f_mean, None, v=pref_v, u=pref_u, return_g_f=False)
        else:
            # since we subtract the two f-values, the correlations between them are flipped, hence the '-' in the last
            # two covariance terms here.
            m_post = self.forward_model(f_mean, f_cov[pref_v, pref_v] + f_cov[pref_u, pref_u] - f_cov[pref_v, pref_u]
                                    - f_cov[pref_u, pref_v], v=pref_v, u=pref_u, return_g_f=False)
        m_post = temper_extreme_probs(m_post)
        not_m_post = 1 - m_post

        return m_post, not_m_post,
