"""
Classification metrics for imbalanced data.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2018.

import numpy as np
import pandas as pd

from sklearn.metrics import auc
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.utils.multiclass import type_of_target

from .util import _check_targets


def zero_division(num, den):
    def _der(num_elt, den_elt):
        if den_elt == 0:
            if num_elt == 0:
                return 1
            else:
                return 9999  # infinity
        else:
            return num_elt/den_elt

    if isinstance(den, (list, np.ndarray)):
        result_vec = [_der(num[i], elt) for i, elt in enumerate(den)]
        return np.array(result_vec)
    else:
        return _der(num, den)


def confusion_matrix_multiclass(y_true, y_pred, labels):
    """
    Compute confusion matrix to evaluate the accuracy of a multiclass
    classification.

    Extract True positive, False positive, False negative and True negative
    from the confusion matrix for multiclass target.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    labels : array, shape [n_labels]
        Labels in the target variable.

    Returns
    -------
    TN, FP, FN, TP : array, shape = [n_classes]
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (fp + fn + tp)

    return tn, fp, fn, tp


def auc_pr(y_true, y_pred_proba):
    """
    Compute area under the Precision-Recall (PR) curve for binary
    classification.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred_proba : array, shape [n_samples]
        Estimated probability of target=1 as returned by a classifier.

    Returns
    -------
    score : float
    """
    y_type = type_of_target(y_true)

    if y_type != "binary":
        raise ValueError("target y_true is not binary.")

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)


def auc_roc(y_true, y_pred_proba):
    """
    Compute area under the Receiver Operating Characteristic (ROC) curve for
    binary classification.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred_proba : array, shape [n_samples]
        Estimated probability of target=1 as returned by a classifier.

    Returns
    -------
    score : float
    """
    y_type = type_of_target(y_true)

    if y_type != "binary":
        raise ValueError("target y_true is not binary.")

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return auc(fpr, tpr)


def balanced_accuracy(y_true, y_pred, func=None, labels=[0, 1], **kwargs):
    r"""
    Compute the balanced classification rate or balanced accuracy
    (:math:`BCR`).

    This metric combines the sensitivity (:math:`TPR`) and specificity
    (:math:`TNR`) metric, and is defined as

    .. math::

        BCR = \frac{1}{2}(TPR + TNR) = \frac{1}{2}\left(\frac{TP}{TP+FN} +
        \frac{TN}{TN + FP}\right).

    For multiclass target, this metric differs from
    ``sklearn.metrics.balance_accuracy_score``, which computes the average of
    recall (sensitivity) obtained on each class. The latter can be computed
    using ``sensitivity(y_true, y_pred, func=np.mean)``.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    func : object (default=None)
        Function to apply to the scores for multiclass targets.

    labels : array, shape [n_labels] (default=[0,1])
        Labels in the target variable.

    kwargs : keyword arguments
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    score : float (if func is not None) or array of float, shape =\
        [n_unique_labels]

    See also
    --------
    sensitivity, specificity

    Notes
    -----
    This metric is suitable for imbalanced data.

    Examples
    --------
    >>> from grmlab.modelling.metrics import balance_accuracy
    >>> y_true = [0, 1, 0, 0, 1, 0]
    >>> y_pred = [0, 1, 0, 0, 0, 1]
    >>> balanced_accuracy(y_true, y_pred)
    0.625
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    if y_type == "multiclass":
        tn, fp, fn, tp = confusion_matrix_multiclass(
            y_true, y_pred, labels=labels)
    else:
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=labels).ravel()

    tpr = zero_division(tp, tp + fn)
    tnr = zero_division(tn, fp + tn)
    bacc = 0.5 * (tpr + tnr)
    if func is not None:
        return func(bacc, **kwargs)
    else:
        return bacc


def balanced_error_rate(y_true, y_pred, func=None, **kwargs):
    """
    Compute the balance error rate (:math:`BER`) or half total error rate
    (:math:`HTER`).

    The balanced error rate is defined as :math:`BER = 1 - BCR`.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    func : object (default=None)
        Function to apply to the scores for multiclass targets.

    kwargs : keyword arguments
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    score : float (if func is not None) or array of float, shape =\
        [n_unique_labels]

    See also
    --------
    balanced_accuracy

    Notes
    -----
    This metric is suitable for imbalanced data.
    """
    return 1.0 - balanced_accuracy(y_true, y_pred, func, **kwargs)


def diagnostic_odds_ratio(y_true, y_pred, func=None, labels=[0, 1], **kwargs):
    r"""
    Compute the diagnostic odds ratio (:math:`DOR`).

    Combined measure of positive and negative likelihood. This measure is
    utilized for estimating the discriminative ability of the test. This metric
    represents the ratio between the positive likelihood ratio (:math:`LR+`) to
    the negative likelihood ratio (:math:`LR-`), and is also expressible in
    terms of the sensitivity (:math:`TPR`) and specificity (:math:`TNR`) as
    follows

    .. math::

        DOR = \frac{LR+}{LR-} = \frac{TPR}{1-TNR} =
        \frac{TP \cdot TN}{FP \cdot FN}.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    func : object (default=None)
        Function to apply to the scores for multiclass targets.

    labels : array, shape [n_labels] (default=[0,1])
        Labels in the target variable.

    kwargs : keyword arguments
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    score : float (if func is not None) or array of float, shape =\
        [n_unique_labels]

    See also
    --------
    sensitivity, specificity

    Notes
    -----
    This metric is suitable for imbalanced data.
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    if y_type == "multiclass":
        tn, fp, fn, tp = confusion_matrix_multiclass(
            y_true, y_pred, labels=labels)
    else:
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=labels).ravel()

    dor = zero_division(tp * tn, fp * fn)

    if func is not None:
        return func(dor, **kwargs)
    else:
        return dor


def discriminant_power(y_true, y_pred, func=None, labels=[0, 1], **kwargs):
    r"""
    Compute the discriminat power (:math:`DP`).

    This metric evaluates how well the classification model distinguishes
    between positive and negative samples. This metric is expressible in terms
    of the sensitivity (:math:`TPR`) and specificitiy (:math:`TNR`), and is
    given by

    .. math::

        DP = \frac{\sqrt{3}}{\pi}\log\left(\frac{TPR}{1-TNR}
        \frac{TNR}{1-TPR}\right).

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    func : object (default=None)
        Function to apply to the scores for multiclass targets.

    labels : array, shape [n_labels] (default=[0,1])
        Labels in the target variable.

    kwargs : keyword arguments
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    score : float (if func is not None) or array of float, shape =\
        [n_unique_labels]

    See also
    --------
    sensitivity, specificity

    Notes
    -----
    This metric is suitable for imbalanced data.
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    if y_type == "multiclass":
        tn, fp, fn, tp = confusion_matrix_multiclass(
            y_true, y_pred, labels=labels)
    else:
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=labels).ravel()

    tpr = zero_division(tp, tp + fn)
    tnr = zero_division(tn, fp + tn)
    c = 0.55132889542179  # sqrt(3) / pi
    dp = c * np.log(zero_division(tpr*tnr, (1.0-tnr) * (1.0-tpr)))

    if func is not None:
        return func(dp, **kwargs)
    else:
        return dp


def false_negative_rate(y_true, y_pred, func=None, **kwargs):
    r"""
    Compute the false negative rate (:math:`FNR`).

    False negative rate or miss rate is the proportion of positive samples that
    were incorrectly classified. This metric complements the sensitivity
    (:math:`TPR`) as follows

    .. math::

        FNR = 1 - TPR = \frac{FN}{FN + TP}.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    func : object (default=None)
        Function to apply to the scores for multiclass targets.

    kwargs : keyword arguments
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    score : float (if func is not None) or array of float, shape =\
        [n_unique_labels]

    See also
    --------
    sensitivity

    Notes
    -----
    This metric is suitable for imbalanced data.
    """
    return 1.0 - sensitivity(y_true, y_pred, func, **kwargs)


def false_positive_rate(y_true, y_pred, func=None, **kwargs):
    r"""
    Compute the false positive rate (:math:`FPR`).

    False positive rate is also called false alarm rate or fallout, and
    it represents the ratio between the incorrectly classified negative samples
    to the total number of negative samples. It is the proportion of the
    negative samples that were incorrectly classified. It is representable
    as the complementary of the specificity (:math:`TNR`) by

    .. math::

        FPR = 1 - TNR = \frac{FP}{FP + TN}.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    func : object (default=None)
        Function to apply to the scores for multiclass targets.

    kwargs : keyword arguments
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    score : float (if func is not None) or array of float, shape =\
        [n_unique_labels]

    See also
    --------
    specificity

    Notes
    -----
    This metric is suitable for imbalanced data.
    """
    return 1.0 - specificity(y_true, y_pred, func, **kwargs)


def geometric_mean(y_true, y_pred, func=None, labels=[0, 1], **kwargs):
    r"""
    Compute the geometric mean (:math:`GM`).

    The goal of all classifiers is to improve sensitivity (:math:`TPR`) without
    sacrificing the specificity (:math:`TNR`). This metric aggregates both
    sensitivity and specificity measures

    .. math::

        GM = \sqrt{TPR \cdot TNR}.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    func : object (default=None)
        Function to apply to the scores for multiclass targets.

    labels : array, shape [n_labels] (default=[0,1])
        Labels in the target variable.

    kwargs : keyword arguments
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    score : float (if func is not None) or array of float, shape =\
        [n_unique_labels]

    See also
    --------
    sensitivity, specificity

    Notes
    -----
    This metric is suitable for imbalanced data.
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    if y_type == "multiclass":
        tn, fp, fn, tp = confusion_matrix_multiclass(
            y_true, y_pred, labels=labels)
    else:
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=labels).ravel()

    tpr = zero_division(tp, tp + fn)
    tnr = zero_division(tn, fp + tn)
    gm = np.sqrt(tpr * tnr)
    if func is not None:
        return func(gm, **kwargs)
    else:
        return gm


def gini(y_true, y_pred_proba):
    r"""
    Compute the Gini index for binary classification.

    This metrics is unfortunately known by many names, e.g. Accuracy Ratio (AR)
    or Sommer's D. The Gini index is defined by

    .. math::

        Gini = 2 \cdot AUCROC - 1.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred_proba : array, shape [n_samples]
        Estimated probability of target=1 as returned by a classifier.

    Returns
    -------
    score : float

    See also
    --------
    auc_roc
    """
    return 2.0 * auc_roc(y_true, y_pred_proba) - 1.0


def negative_likelihood(y_true, y_pred, func=None, labels=[0, 1], **kwargs):
    r"""
    Compute the negative likelihood measure (:math:`LR-`).

    This metric measures how much the odds of the event decreases when
    the test is negative. This metric is expressible in terms of
    the sensitivity (:math:`TPR`) and specificity (:math:`TNR`).

    .. math::

        LR- = \frac{1-TPR}{TNR}.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    func : object (default=None)
        Function to apply to the scores for multiclass targets.

    labels : array, shape [n_labels] (default=[0,1])
        Labels in the target variable.

    kwargs : keyword arguments
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    score : float (if func is not None) or array of float, shape =\
        [n_unique_labels]

    See also
    --------
    sensitivity, specificity

    Notes
    -----
    This metric is suitable for imbalanced data.
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    if y_type == "multiclass":
        tn, fp, fn, tp = confusion_matrix_multiclass(
            y_true, y_pred, labels=labels)
    else:
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=labels).ravel()

    tpr = zero_division(tp, tp + fn)
    tnr = zero_division(tn, fp + tn)
    lr = zero_division(1.0 - tpr, tnr)
    if func is not None:
        return func(lr, **kwargs)
    else:
        return lr


def negative_lift(y_true, y_pred, func=None, labels=[0, 1], **kwargs):
    r"""
    Compute negative precision lift expressed as the ratio of negative
    predictive value to the ratio of negative samples. This can be interpreted
    as the improvement over random sampling.

    .. math::

        Lift- = \frac{NPV}{RN}, \quad RN = \frac{TN + FP}
        {TP + TN + FP + FN}.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    func : object (default=None)
        Function to apply to the scores for multiclass targets.

    labels : array, shape [n_labels] (default=[0,1])
        Labels in the target variable.

    kwargs : keyword arguments
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    score : float (if func is not None) or array of float, shape =\
        [n_unique_labels]

    See also
    --------
    negative_predictive_value
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    if y_type == "multiclass":
        tn, fp, fn, tp = confusion_matrix_multiclass(
            y_true, y_pred, labels=labels)
    else:
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=labels).ravel()

    npv = zero_division(tn, fn + tn)
    rn = zero_division(tn + fp, tp + tn + fp + fn)
    nlift = zero_division(npv, rn)
    if func is not None:
        return func(nlift, **kwargs)
    else:
        return nlift


def negative_predictive_value(
        y_true, y_pred, func=None, labels=[0, 1], **kwargs):
    r"""
    Compute the inverse precision or negative predictive value (:math:`NPV`).

    This metric represents the proportion of correctly classified negative
    samples to the total number of negative predicted samples.

    .. math::

        NPV = \frac{TN}{FN + TN}.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    func : object (default=None)
        Function to apply to the scores for multiclass targets.

    labels : array, shape [n_labels] (default=[0,1])
        Labels in the target variable.

    kwargs : keyword arguments
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    score : float (if func is not None) or array of float, shape =\
        [n_unique_labels]
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    if y_type == "multiclass":
        tn, fp, fn, tp = confusion_matrix_multiclass(
            y_true, y_pred, labels=labels)
    else:
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=labels).ravel()

    npv = zero_division(tn, fn + tn)
    if func is not None:
        return func(npv, **kwargs)
    else:
        return npv


def positive_likelihood(y_true, y_pred, func=None, labels=[0, 1], **kwargs):
    r"""
    Compute the positive likelihood measure (:math:`LR+`).

    Positive likelihood measures how much the odds of the event increases
    when a test is positive. This metric is expressible in terms of the
    sensitivity (:math:`TPR`) and specificity (:math:`TNR`).

    .. math::

        LR+ = \frac{TPR}{1-TNR}.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    func : object (default=None)
        Function to apply to the scores for multiclass targets.

    labels : array, shape [n_labels] (default=[0,1])
        Labels in the target variable.

    kwargs : keyword arguments
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    score : float (if func is not None) or array of float, shape =\
        [n_unique_labels]

    See also
    --------
    sensitivity, specificity

    Notes
    -----
    This metric is suitable for imbalanced data.
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    if y_type == "multiclass":
        tn, fp, fn, tp = confusion_matrix_multiclass(
            y_true, y_pred, labels=labels)
    else:
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=labels).ravel()

    tpr = zero_division(tp, tp + fn)
    tnr = zero_division(tn, fp + tn)

    lp = zero_division(tpr, 1.0 - tnr)

    if func is not None:
        return func(lp, **kwargs)
    else:
        return lp


def positive_lift(y_true, y_pred, func=None, labels=[0, 1], **kwargs):
    r"""
    Compute positive precision lift expressed as the ratio of positive
    predictive value to the ratio of positive samples. This can be interpreted
    as the improvement over random sampling.

    .. math::

        Lift+ = \frac{PPV}{RP}, \quad RP = \frac{TP + FN}
        {TP + TN + FP + FN}.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    func : object (default=None)
        Function to apply to the scores for multiclass targets.

    labels : array, shape [n_labels] (default=[0,1])
        Labels in the target variable.

    kwargs : keyword arguments
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    score : float (if func is not None) or array of float, shape =\
        [n_unique_labels]

    See also
    --------
    negative_predictive_value
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    if y_type == "multiclass":
        tn, fp, fn, tp = confusion_matrix_multiclass(
            y_true, y_pred, labels=labels)
    else:
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=labels).ravel()

    ppv = zero_division(tp, fp + tp)
    rp = zero_division(tp + fn, tp + tn + fp + fn)

    plift = zero_division(ppv, rp)

    if func is not None:
        return func(plift, **kwargs)
    else:
        return plift


def positive_predictive_value(
        y_true, y_pred, func=None, labels=[0, 1], **kwargs):
    r"""
    Compute the precision or positive predictive value (:math:`PPV`).

    This metric represents the proportion of correctly classified positive
    samples to the total number of positive predicted samples.

    .. math::

        PPV = \frac{TP}{FP + TP}.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    func : object (default=None)
        Function to apply to the scores for multiclass targets.

    labels : array, shape [n_labels] (default=[0,1])
        Labels in the target variable.

    kwargs : keyword arguments
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    score : float (if func is not None) or array of float, shape =\
        [n_unique_labels]
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    if y_type == "multiclass":
        tn, fp, fn, tp = confusion_matrix_multiclass(
            y_true, y_pred, labels=labels)
    else:
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=labels).ravel()

    ppv = zero_division(tp, fp + tp)
    if func is not None:
        return func(ppv, **kwargs)
    else:
        return ppv


def sensitivity(y_true, y_pred, func=None, labels=[0, 1], **kwargs):
    r"""
    Compute the sensitivity or true positive rate (:math:`TPR`).

    True positive rate, hit rate or recall, of a classifier represents
    the positive correctly classified samples to the total number of positive
    samples.

    .. math::

        TPR = \frac{TP}{TP + FN}.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    func : object (default=None)
        Function to apply to the scores for multiclass targets.

    labels : array, shape [n_labels] (default=[0,1])
        Labels in the target variable.

    kwargs : keyword arguments
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    score : float (if func is not None) or array of float, shape =\
        [n_unique_labels]
        sensitivity, which is a number between [0, 1].

    Notes
    -----
    This metric is suitable for imbalanced data.

    Examples
    --------
    >>> import numpy as np
    >>> from grmlab.modelling.metrics import sensitivity
    >>> y_true = [0, 1, 2, 0, 2, 1, 2]
    >>> y_pred = [0, 1, 1, 1, 2, 1, 2]
    >>> sensitivity(y_true, y_pred, func=None)
    array([0.5       , 1.        , 0.66666667])
    >>> sensitivity(y_true, y_pred, func=np.mean)
    0.7222222222222222
    >>> sensitivity(y_true, y_pred, func=np.average, weights=[2, 2, 3])
    0.7142857142857143
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    if y_type == "multiclass":
        tn, fp, fn, tp = confusion_matrix_multiclass(
            y_true, y_pred, labels=labels)
    else:
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=labels).ravel()

    tpr = zero_division(tp, tp + fn)
    if func is not None:
        return func(tpr, **kwargs)
    else:
        return tpr


def specificity(y_true, y_pred, func=None, labels=[0, 1], **kwargs):
    r"""
    Compute the specificity or true negative rate (:math:`TNR`).

    True negative rate or inverse recall is expressed as the ratio of the
    correctly classified negative samples to the total number of negative
    samples.

    .. math::

        TNR = \frac{TN}{FP + TN}.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    func : object (default=None)
        Function to apply to the scores for multiclass targets.

    labels : array, shape [n_labels] (default=[0,1])
        Labels in the target variable.

    kwargs : keyword arguments
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    score : float (if func is not None) or array of float, shape =\
        [n_unique_labels]
        specificity, which is a number between [0, 1].

    Notes
    -----
    This metric is suitable for imbalanced data.

    Examples
    --------
    >>> import numpy as np
    >>> from grmlab.modelling.metrics import specificity
    >>> y_true = [0, 1, 2, 0, 2, 1, 2]
    >>> y_pred = [0, 1, 1, 1, 2, 1, 2]
    >>> specificity(y_true, y_pred)
    array([1. , 0.6, 1. ])
    >>> specificity(y_true, y_pred, func=np.mean)
    0.8666666666666667
    >>> specificity(y_true, y_pred, func=np.average, weights=[2, 2, 3])
    0.8857142857142858
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    if y_type == "multiclass":
        tn, fp, fn, tp = confusion_matrix_multiclass(
            y_true, y_pred, labels=labels)
    else:
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=labels).ravel()

    tnr = zero_division(tn, fp + tn)
    if func is not None:
        return func(tnr, **kwargs)
    else:
        return tnr


def youden_index(y_true, y_pred, func=None, labels=[0, 1], **kwargs):
    r"""
    Compute the Youden's index (:math:`YI`).

    Returns Youden's index or Bookmaker Informedness metric, which is
    a well-known diagnostic test. It evaluates the discriminative power of the
    test. The formula of Youden's index combines the sensitivity (:math:`TPR`)
    and specificity (:math:`TNR`) given

    .. math::

        YI = TPR + TNR - 1

    The :math:`YI` metric is ranged from zero when the test is poor to one
    which represents a perfect diagnostic test.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    func : object (default=None)
        function to apply to the scores for multiclass targets.

    labels : array, shape [n_labels] (default=[0,1])
        Labels in the target variable.

    kwargs : keyword arguments
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    score : float (if func is not None) or array of float, shape =\
        [n_unique_labels]

    See also
    --------
    sensitivity, specificity

    Notes
    -----
    This metric is suitable for imbalanced data.
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    if y_type == "multiclass":
        tn, fp, fn, tp = confusion_matrix_multiclass(
            y_true, y_pred, labels=labels)
    else:
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=labels).ravel()

    tpr = zero_division(tp, tp + fn)
    tnr = zero_division(tn, fp + tn)
    yi = tpr + tnr - 1
    if func is not None:
        return func(yi, **kwargs)
    else:
        return yi


_METRICS = ("balanced_accuracy", "balanced_error_rate",
            "diagnostic_odds_ratio", "discriminant_power", "fnr", "fpr",
            "geometric_mean", "negative_likelihood", "positive_likelihood",
            "tpr", "tnr", "youden")


_METRICS_SHORTCUTS = ("bcr", "ber", "dor", "dp", "fnr", "fpr", "gm", "ln",
                      "lp", "tpr", "tnr", "yi")


def _check_metric(metric):
    """Check if metric is valid."""
    if metric in _METRICS:
        return metric
    elif metric in _METRICS_SHORTCUTS:
        return next(_METRICS[i] for i, m in enumerate(_METRICS_SHORTCUTS)
                    if metric == m)
    else:
        raise ValueError("{} is not a valid imbalanced metric.".format(metric))


def imbalanced_classification_report(y_true, y_pred, metrics=None,
                                     binary=False, target_names=None,
                                     func=None, output_dict=False,
                                     labels=[0, 1], **kwargs):
    """
    Build a complete report with all or several imbalanced classification
    metrics.

    Available metrics are::

        metrics = ["balanced_accuracy", "balanced_error_rate",
        "diagnostic_odds_ratio", "discriminant_power", "false_negative_rate",
        "false_positive_rate", "geometric_mean", "negative_likelihood",
        "positive_likelihood", "sensitivity", "specificity", "youden"]

    or their corresponding shortcuts::

        metrics = ["bcr", "ber", "dor", "dp", "fnr", "fpr", "gm", "ln", "lp",
        "tpr", "tnr", "yi"]

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    metrics : array-like (default=None)
        Imbalanced metrics for reporting. If None, all metrics are included.

    binary : bool (default=False)
        Whether to report only the positive class, i.e. class = 1 when target
        is binary. Otherwise the report returns metrics for class 0 and 1.

    target_names : list of strings, (default=None)
        Optional display column names matching the labels (same order).

    func : object (default=None)
        Function to apply to the scores for multiclass targets.

    output_dict : bool (default=False)
        If True, return output as dict, otherwise return pandas.DataFrame.

    labels : array, shape [n_labels] (default=[0,1])
        Labels in the target variable.

    kwargs : keyword arguments
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    report : dict or pandas.DataFrame

    Examples
    --------
    see Examples section.
    """

    # Check if all metrics are valid imbalanced metrics
    if metrics is not None:
        _metrics = [_check_metric(metric) for metric in metrics]
    else:
        _metrics = _METRICS

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    if y_type == "multiclass" or not binary:
        tn, fp, fn, tp = confusion_matrix_multiclass(
            y_true, y_pred, labels=labels)
    else:
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=labels).ravel()

    # compute sensitivity (tpr) and specificity (tnr)
    tpr = zero_division(tp, tp + fn)
    tnr = zero_division(tn, fp + tn)

    report_dict = {}

    for metric in _metrics:
        if metric == "balanced_accuracy":
            value = 0.5 * (tpr + tnr)
        elif metric == "balanced_error_rate":
            value = 1.0 - 0.5 * (tpr + tnr)
        elif metric == "diagnostic_odds_ratio":
            value = zero_division(tp * tn, fp * fn)
        elif metric == "discriminant_power":
            c = 0.55132889542179
            value = c * np.log(zero_division(tpr*tnr, (1.0 - tnr)*(1.0 - tpr)))
        elif metric == "fnr":
            value = 1.0 - tpr
        elif metric == "fpr":
            value = 1.0 - tnr
        elif metric == "geometric_mean":
            value = np.sqrt(tpr * tnr)
        elif metric == "negative_likelihood":
            value = zero_division(1.0 - tpr, tnr)
        elif metric == "positive_likelihood":
            value = zero_division(tpr, 1.0 - tnr)
        elif metric == "tpr":
            value = tpr
        elif metric == "tnr":
            value = tnr
        elif metric == "youden":
            value = tpr + tnr - 1.0

        if func is not None:
            report_dict[metric] = func(value, **kwargs)
        else:
            report_dict[metric] = value

    if output_dict:
        return report_dict
    else:
        report_df = pd.DataFrame.from_dict(
            report_dict, orient='index').sort_index()

        if target_names is None:
            if binary:
                report_df.columns = [1]
        else:
            report_df.columns = target_names

        return report_df


def binary_classification_report(y_true, y_pred, y_pred_proba=None,
                                 metrics=None, output_dict=True,
                                 labels=[0, 1]):
    """
    Builds a complete report with all or several metrics for binary
    classification.

    Parameters
    ----------
    y_true : array, shape [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape [n_samples]
        Estimated targets as returned by a classifier.

    y_pred_proba : array, shape [n_samples] or None (default=None)
        Estimated probability of target=1 as returned by a classifier.

    metrics : array-like (default=None)
        Metrics for reporting. If None, all metrics are included.

    output_dict : boolean (default=True)
        If True, returns a dict. If False, returns a pandas dataframe.

    labels : array, shape [n_labels] (default=[0,1])
        Labels in the target variable.
    """
    _BINARY_METRICS = [
        "tp", "tn", "fp", "fn", "tpr", "tnr", "fpr", "fnr",
        "youden", "ppv", "npv", "lift_p", "lift_n", "gini", "kappa", "mcc",
        "log_loss", "accuracy", "balanced_accuracy", "balanced_accuracy",
        "balanced_error_rate", "diagnostic_odds_ratio", "discriminant_power",
        "false_negative_rate", "false_positive_rate", "geometric_mean",
        "negative_likelihood", "positive_likelihood", "default_rate"]

    _CM_METRICS = [
        "tp", "tn", "fp", "fn", "tpr", "tnr", "fpr", "fnr", "youden",
        "accuracy", "ppv", "npv", "lift_p", "lift_n", "balanced_accuracy",
        "balanced_error_rate", "diagnostic_odds_ratio", "geometric_mean",
        "discriminant_power", "negative_likelihood", "positive_likelihood"]

    # Check if all metrics are valid imbalanced metrics
    if metrics is not None:
        _metrics = []
        for metric in metrics:
            if metric not in _BINARY_METRICS:
                raise ValueError("{} is not a valid binary classification "
                                 "metric.".format(metric))
            else:
                _metrics.append(metric)
    else:
        _metrics = _BINARY_METRICS

    # check if pred_proba is provided. Required for gini and log_loss
    if any(metric in _metrics for metric in ["gini", "log_loss"]) and (
            y_pred_proba is None):
        raise ValueError("y_pred_proba must be provided for 'gini' and "
                         "'log_loss' metrics.")

    _, y_true, y_pred = _check_targets(y_true, y_pred)

    n_samples = len(y_true)

    if any([metric in _CM_METRICS] for metric in _metrics):
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=labels).ravel()

    # compute sensitivity (tpr) and specificity (tnr)
    tpr = tp / (tp + fn)
    tnr = tn / (fp + tn)

    report_dict = {}

    for metric in _metrics:
        if metric == "tp":
            value = tp
        elif metric == "tn":
            value = tn
        elif metric == "fp":
            value = fp
        elif metric == "fn":
            value = fn
        elif metric == "tpr":
            value = tpr
        elif metric == "tnr":
            value = tnr
        elif metric == "fpr":
            value = zero_division(fp, fp + tn)
        elif metric == "fnr":
            value = zero_division(fn, fn + tp)
        elif metric == "youden":
            value = tpr + tnr - 1.0
        elif metric == "ppv":
            value = zero_division(tp, tp + fp)
        elif metric == "npv":
            value = zero_division(tn, tn + fn)
        elif metric == "lift_p":
            ppv = zero_division(tp, tp + fp)
            value = zero_division(ppv, (tp + fn) / n_samples)
        elif metric == "lift_n":
            npv = zero_division(tn, tn + fn)
            value = zero_division(npv, (tn + fp) / n_samples)
        elif metric == "gini":
            value = gini(y_true, y_pred_proba)
        elif metric == "kappa":
            value = cohen_kappa_score(y_true, y_pred)
        elif metric == "mcc":
            value = matthews_corrcoef(y_true, y_pred)
        elif metric == "log_loss":
            value = log_loss(y_true, y_pred_proba)
        elif metric == "accuracy":
            value = (tp + tn) / n_samples
        elif metric == "balanced_accuracy":
            value = 0.5 * (tpr + tnr)
        elif metric == "balanced_error_rate":
            value = 1.0 - 0.5 * (tpr + tnr)
        elif metric == "diagnostic_odds_ratio":
            value = zero_division(tp * tn, fp * fn)
        elif metric == "discriminant_power":
            c = 0.55132889542179
            value = c * np.log(zero_division(tpr*tnr, (1.0 - tnr)*(1.0 - tpr)))
        elif metric == "geometric_mean":
            value = np.sqrt(tpr * tnr)
        elif metric == "negative_likelihood":
            value = zero_division(1.0 - tpr, tnr)
        elif metric == "positive_likelihood":
            value = zero_division(tpr, 1.0 - tnr)
        elif metric == "default_rate":
            value = np.nanmean(y_true)

        report_dict[metric] = value

    if output_dict:
        return report_dict
    else:
        report_df = pd.DataFrame.from_dict(
            report_dict, orient='index').sort_index()

        return report_df
