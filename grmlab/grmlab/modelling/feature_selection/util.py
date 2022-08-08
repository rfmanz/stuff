"""
Auxiliary functions for feature selection.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import numpy as np

from sklearn.utils import check_X_y


def condition_number(X):
	"""
	Compute the condition number of matrix.

	Compute condition number of :math:`\kappa(X^TX)`, where :math:`X` is the
	resulting matrix with only those features selected by the feature selection
	algorithm. The condition numbers is computed as :math:`\kappa(X) =
	\sqrt{|\lambda_{max}(X) / \lambda_{min}(X)|}`, where :math:`\lambda_{max}(X)` 
	and :math:`\lambda_{min}(X)`, are the maximum and minimum eigenvalue, 
	respectively.

	Parameters
	----------
	X : array of shape [n_samples, n_features]
		The input samples.

	Returns
	-------
	k : float
		The condition number of the matrix.
	"""
	eigenvalues = np.linalg.eigvals(np.dot(X.T, X))
	max_eigen, min_eigen = eigenvalues.max(), eigenvalues.min()
	return np.sqrt(abs(max_eigen / min_eigen))


def correlation_analysis(X):
	"""
	Compute correlation matrix and correlation distribution.

	Parameters
	----------
	X : array of shape [n_samples, n_features]
		The input samples.

	Returns
	-------
	corr_dist : array
		The statistics max, min, mean, percentile25, median and percentile75.
	"""
	n_samples, n_features = X.shape

	corr = np.corrcoef(X.T)
	corr_array = np.asarray([corr[i, j]
		for i in range(n_features) for j in range(i)])

	max_corr = np.max(corr_array)
	min_corr = np.min(corr_array)
	mean_corr = np.mean(corr_array)
	[p25_corr, p50_corr, p75_corr] = np.percentile(corr_array, [25, 50, 75])

	return max_corr, min_corr, mean_corr, p25_corr, p50_corr, p75_corr



def corrcoef_X_y(X, y):
	"""
	Compute correlation coefficient between matrix X and array y.

	Parameters
	----------
	X : {array-like, sparse matrix}, shape = [n_samples, n_features]
		The input samples.

	y : array-like, shape = [n_samples]
		The target values.

	Returns
	-------
	corr : array
		The correlation between each column of X and array y.
	"""

	# check X, y consistency
	X, y = check_X_y(X, y, "csc")

	n = X.shape[1]

	Xs = X - X.mean(axis=0)
	ys = y.sum()
	y -= ys / n
	return np.dot(y, Xs) / np.sqrt(ys * np.einsum('ij,ij->j', Xs, Xs))
