"""
Robust Logistic regression algorithm.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import numbers

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target

from ...core.exceptions import NotFittedException


def empirical_risk(X, y, classifier):
	"""
	Compute the empirical risk function given by

	.. math::

		R = \\frac{1}{n} \\sum_{i=1}^{n} (1 + \\exp(-y_i \\text{
		decision_function(classifier)}))^{-1}

	Parameters
	----------
	X : {array-like, sparse matrix}, shape = [n_samples, n_features]
		The training input samples.

	y : array-like, shape = [n_samples]
		The target values. It must be a **binary** array.

	classifier : object
		A supervised learning estimator with method ``decision_function``.

	Returns
	-------
	r : float
	"""
	mask = (y == 1)
	y_bin = np.ones(y.shape, dtype=X.dtype)
	y_bin[~mask] = -1

	# scores
	if not hasattr(classifier, "decision_function"):
		raise TypeError("classifier must contain method decision_function.")

	scores = classifier.decision_function(X)
	emprisk = 1.0 / (1.0 + np.exp(-np.multiply(y_bin, scores)))
	return np.sum(emprisk) / len(y_bin)


class RobustLogisticRegression(BaseEstimator):
	"""
	Robust logistic regression classifier.

	Logistic regression maximizing the linear correlation statistic. This
	algorithm is suitable for binary classification problems with imbalanced
	target classes.

	Parameters
	----------
	pairwise : boolean (default=False)
		Whether to consider pairwise relations. These are second order relations
		with the target.

	C : float (default=0.1)
		Regularization strength; must be a positive float. Larger values specify
		stronger pairwise relations.

	fit_intercept : boolean (default=True)
		Specifies if a constant should be added to the decision function.

	alpha : float or None (default=1.0)
		Upper bound for the :math:`L_2`-norm of the coefficients. If None,
		'alphaa' is set to :math:`\\sqrt{n}`, where :math:`n` is the number of
		features + intercept (if ``fit_intercept`` = True).

	beta : float (default=1.0)
		Upper bound for the :math:`L_2`-norm for the pairwise coefficients.

	verbose : int or boolean (default=False)
		Controls verbosity of output.

	Attributes
	----------
	coef_ : array, shape = [1, n_features]
		Coefficient of the features in the decision function.

	coef_q_ : array, shape = [1, n_features (n_features - 1)]
		Coefficient of pairwise relations among features in the decision
		function.

	intercept_ : float
		Intercept added to the decision function.
	"""
	def __init__(self, fit_intercept=True, pairwise=False, C=0.1, alpha=1.0,
		beta=1.0, verbose=False):
		# parameters
		self.C = C
		self.alpha = alpha
		self.beta = beta
		self.fit_intercept = fit_intercept
		self.pairwise = pairwise
		self.verbose = verbose

		# regression attributes
		self.coef_ = None
		self.intercept_ = None
		self.coef_q_ = None

		# flag
		self._is_fitted = False

	def fit(self, X, y):
		"""
		Fit the model according to the given training data.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape = [n_samples, n_features]
			The training input samples.

		y : array-like, shape = [n_samples]
			The target values. It must be a binary array, otherwise an exception
			is raised.

		Returns
		-------
		self : object
		"""
		if not isinstance(self.C, numbers.Number) or self.C < 0:
			raise ValueError("C term must be positive;"
				" got (C={})".format(self.C))

		if not isinstance(self.alpha, numbers.Number) or self.alpha < 0:
			raise ValueError("alpha term must be positive;"
				" got (alpha={})".format(self.alpha))

		if not isinstance(self.beta, numbers.Number) or self.beta < 0:
			raise ValueError("beta term must be positive;"
				" got (beta={})".format(self.radius_2))

		# check X, y consistency
		X, y = check_X_y(X, y, "csc")

		# check whether y is binary
		n_classes = len(np.unique(y))
		if n_classes != 2:
			raise ValueError("this solver needs samples of two classes: 0 and "
				"1, but the data contains {} classes".format(n_classes))

		# a constant array of 0 or 1 is considered binary
		if type_of_target(y) is not "binary":
			raise ValueError("target y must be a binary array.")

		n_samples, n_features = X.shape

		# mask labels
		mask = (y == 1)
		y_bin = np.ones(y.shape, dtype=X.dtype)
		y_bin[~mask] = -1

		# solve linear relation
		if self.fit_intercept:
			c = np.dot(np.c_[X, np.ones(n_samples)].T, y_bin)
		else:
			c = np.dot(X.T, y_bin)		
		self.coef_ = c / np.sqrt(np.dot(c, c))

		# upper bound linear and quadratic coefficients
		if self.alpha is None:
			if self.fit_intercept:
				r = np.sqrt(n_features + 1)
			else:
				r = np.sqrt(n_features)

			self.coef_ *= r
		else:
			self.coef_ *= np.sqrt(self.alpha)

		if self.fit_intercept:
			self.intercept_ = self.coef_[-1]
			self.coef_ = self.coef_[:-1]

		# solve quadratic relation (pairwise)
		if self.pairwise or self.C != 0:
			c = self.C * np.asarray([np.dot(X[:, j]*X[:, k], y_bin) for j in 
				range(n_features) for k in range(j+1, n_features)])
			self.coef_q_ = c / np.sqrt(np.dot(c,c))
			self.coef_q_ *= np.sqrt(self.beta)

		self._is_fitted = True

		return self

	def decision_function(self, X):
		"""
		Predict confidence scores for samples.

		The confidence score for a sample is the signed distance of that sample
		to the hyperplane.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape = [n_samples, n_features]
			The training input samples.

		Returns
		-------
		scores : array of shape = [n_samples].
		"""
		if not self._is_fitted:
			raise NotFittedException(self)

		n_features = self.coef_.shape[0]

		if X.shape[1] != n_features:
			raise ValueError("X has {} features per sample; expecting {}".format(
				X.shape[1], n_features))

		scores = np.dot(X, self.coef_)

		if self.fit_intercept:
			scores += self.intercept_

		if self.pairwise:
			W = np.array([X[:, j]*X[:, k] for j in range(n_features) 
				for k in range(j+1, n_features)])
			scores += np.dot(W.T, self.coef_q_)

		return scores

	def predict(self, X):
		"""
		Predict class label for samples in X.

		Parameters
		----------
		X : array of shape [n_samples, n_features]
		    The input samples.

		Returns
		-------
		y : array of shape [n_samples]
		    The predicted target values.
		"""
		if not self._is_fitted:
			raise NotFittedException(self)

		y_pred = np.zeros(X.shape[0])
		y_prob = self.predict_proba(X)[:, 1]
		y_pred[y_prob > 0.5] = 1

		return y_pred.astype(np.int)

	def predict_proba(self, X):
		"""
		Predict class probabilities for X.

		Parameters
		----------
		X : array-like or sparse matrix, shape = [n_samples, n_features]
		    The input samples.

		Returns
		-------
		p : array of shape = [n_samples]
		    The class probabilities of the input samples.
		"""
		probabilities = np.zeros((X.shape[0], 2))
		probabilities[:, 1] = 1.0 / (1 + np.exp(-self.decision_function(X)))
		probabilities[:, 0] = 1.0 - probabilities[:, 1]

		return probabilities

	def predict_log_proba(self, X):
		"""
		Log of probability estimates.

		Parameters
		----------
		X : array-like or sparse matrix, shape = [n_samples, n_features]
		    The input samples.

		Returns
		-------
		p : array of shape = [n_samples]
		    The logarithm of class probabilities of the input samples.
		"""
		return np.log(self.predict_proba(X))

	def score(self, X, y):
		"""
		Returns the mean accuracy on the given test data and labels.

		Parameters
		----------
		X : array-like or sparse matrix, shape = [n_samples, n_features]
		    The input samples.

		Returns
		-------
		score : float
			Mean accuracy of self.predict(X) wrt. y.
		"""
		from sklearn.metrics import accuracy_score
		return accuracy_score(y, self.predict(X))