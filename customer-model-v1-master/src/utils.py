import numpy as np


def scale_scores(predictions):
    """
    Convert probability to score.
    """
    try:
        assert (
            (predictions >= 0) & (predictions <= 1)
        ).all(), "probability must be in range [0,1]"
    except AssertionError:
        raise

    # Formula parameters
    ALPHA = 631.9455383610933
    BETA = 30.812519272450654

    # Minimum and maximum values for validation
    MINIMUM = 300
    MAXIMUM = 850

    score = np.minimum(
        np.maximum(np.log(predictions / (1 - predictions)) * BETA + ALPHA, MINIMUM),
        MAXIMUM,
    )

    return score


def build_score_coefficients(pred):
    """
    For converting probability to score ranging from 0 to 100 using this formula:

    scores = log(preds / (1 - preds)) * a + b

    Where a and b are:
    a = 100 / (max - min)
    b = - (100 * min) / max

    Where max and min are: 
    max = max(log(preds / (1 - preds)))
    min = min(log(preds / (1 - preds)))
    
    
    call a, b = build_score_coefficients(pred) 
    """
    scores = np.log(pred / (1 - pred))
    s_max = scores.max()
    s_min = scores.min()
    a = 100 / (s_max - s_min)
    b = (100 * s_min) / (s_min - s_max)
    return a, b

