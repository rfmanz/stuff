import pytest

from sklearn.utils.estimator_checks import check_estimator

from pytraining import TemplateEstimator
from pytraining import TemplateClassifier
from pytraining import TemplateTransformer


@pytest.mark.parametrize(
    "estimator",
    [TemplateEstimator(), TemplateTransformer(), TemplateClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
