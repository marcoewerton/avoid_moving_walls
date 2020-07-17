import numpy as np
import scipy as sp
from scipy.interpolate import interp1d


class ProMP:

    def __init__(self, demonstrations, number_features):
        self.max_length = 200  # used to time-align trajectories
        self._number_features = number_features  # Number of Gaussian basis functions
        self._regularizer = 1e-11

        DOFS = demonstrations[0].shape[1]  # The number of columns of each ndarray in the list demonstrations is the number of DoFs.

        #  interpolate all arrays to the same length
        for i in range(len(demonstrations)):
            old_x = np.linspace(0, 1, demonstrations[i].shape[0])  # Returns demonstrations[i].shape[0] numbers from 0 to 1, including 0 and 1. Here, demonstrations[i].shape[0] is the number of positions along trajectory index i.
            new_x = np.linspace(0, 1, self.max_length)
            interp = interp1d(old_x, np.transpose(demonstrations[i]))
            demonstrations[i] = interp(new_x)  # Makes all trajectories have the same length as the self.max_length.
        # After this loop, each ndarray demonstrations[i] has shape (DOFS, T)

        #  learn the model of the trajectories
        # For explanation purposes: N = self._number_features
        # For explanation purposes: m = len(demonstrations)
        self.weight_matrix = np.zeros((self._number_features*DOFS, len(demonstrations)), dtype="float")  # (N*DOFS, m)
        feature_matrix = self.feature_function(self.max_length)  # (T, N)
        for i in range(len(demonstrations)):
            self.weight_matrix[:, i] = self.linear_ridge_regression(feature_matrix, demonstrations[i]).reshape((np.size(self.weight_matrix, 0), ))
        # In order to assign the output of linear_ridge_regression to one of the columns of self.weight_matrix, the output of linear_ridge_regression is first transformed into a rank 1 array.

        self.mu_w = sp.mean(self.weight_matrix, axis=1)
        self.mu_w = self.mu_w.reshape((self.mu_w.size, 1), order='F')
        self.Sigma_w = np.cov(self.weight_matrix)  # (N*DOFs, N*DOFs)
        self.Sigma_w_flat = np.diag(self.Sigma_w)
        self.Sigma = np.identity(self.Sigma_w.shape[0]) * 1e-5  # Matrix used by the sampling scheme of RWPO to learn the relevance

        self.block_PSI = sp.linalg.block_diag(*([feature_matrix] * DOFS)) # shape = (DOFS*self.max_length, DOFS*number_features)

    def feature_function(self, T):
        #  Gaussian basis function
        def feature(x, y): return np.exp(-0.5*(y-(x-1)*T/(self._number_features-1))**2/T)  # exp(-0.5*(y - ((x-1)*T/(N-1)))^2/T)
        # x is the basis function index.
        # y is the time step.
        # The first basis function is centered at 0.
        # The last basis function is centered at T.

        feature_matrix = np.zeros((T, self._number_features), dtype="float")  # (T, N)

        for t in range(1, T + 1):
            for n in range(1, self._number_features+1):
                feature_matrix[t-1, n-1] = feature(n, t)

        normalizer = np.sum(feature_matrix, axis=1)
        feature_matrix = np.divide(feature_matrix, normalizer.reshape(np.size(normalizer, 0), 1))

        return feature_matrix  # (T, N) matrix of normalized Gaussian basis functions

    def linear_ridge_regression(self, feature_input, target_data):  # feature_input is a (T, N) matrix. target_data is a (T, DOFS) matrix.
        product = np.dot(np.transpose(feature_input), feature_input)
        weights = np.dot(np.dot(np.linalg.inv((product + self._regularizer*np.identity(np.size(product, 0)))), np.transpose(feature_input)), np.transpose(target_data))

        weights = weights.reshape((np.size(weights), 1), order='F')

        return weights  # (N*DOFS, 1). [all weights for DoF1; all weights for DoF2; all weights for DoF3; ...]

    # Returns samples from a learnt ProMP
    def sample(self, n_samples, weights=None, rel=False):
        if weights is None:
            # Sample from the ProMP
            # shape = (n_samples, dofs*num_gaba)
            weight_samples = \
                np.random.multivariate_normal(self.mu_w.reshape((self.mu_w.size,)),
                                              self.Sigma if rel else self.Sigma_w, n_samples)
            # shape = (n_samples, dofs*max_length)
            traj_samples = np.dot(weight_samples, np.transpose(self.block_PSI))
        else:
            weight_samples = \
                np.random.multivariate_normal(self.mu_w.reshape((self.mu_w.size,)), weights, n_samples)  # weights here is a covariance matrix
            traj_samples = \
                np.dot(weight_samples, np.transpose(self.block_PSI))

        return weight_samples, traj_samples