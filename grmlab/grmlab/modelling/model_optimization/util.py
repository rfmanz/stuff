"""
Model optimizer parameters generator.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

from ..._thirdparty.hyperopt.hp import choice
from ..._thirdparty.hyperopt.hp import lognormal
from ..._thirdparty.hyperopt.hp import normal
from ..._thirdparty.hyperopt.hp import randint
from ..._thirdparty.hyperopt.hp import uniform


def _add_distribution(name, dist, *args):
    """
    Check whether arguments for each distribution are correct and return
    hyperopt distribution instance.
    """
    if dist not in ["choice", "lognormal", "normal", "randint", "uniform"]:
        raise ValueError("distribution {} is not supported.".format(name))

    if dist is "choice":
        if not isinstance(*args, (list, tuple)):
            raise ValueError("choice requires a list or tuple with "
                             "elements to inspection.")
        else:
            return choice(name, *args)
    elif dist in ("lognormal", "normal"):
        if len(args) != 2:
            raise ValueError("distribution {} requires two arguments, mean "
                             "and standard deviation.".format(name))
        elif args[1] < 0:
            raise ValueError("standard deviation must be positive.")
        elif dist is "lognormal":
            return lognormal(name, *args)
        elif dist is "normal":
            return normal(name, *args)
    elif dist is "randint":
        if len(args) != 1:
            raise ValueError("distribution randint only accepts one argument.")
        elif args[0] < 0:
            raise ValueError("upper value must be positive.")
        else:
            return randint(name, *args)
    elif dist is "uniform":
        if len(args) != 2:
            raise ValueError("distribution uniform requires two arguments, "
                             "low and high value.")
        elif args[0] > args[1]:
            raise ValueError("high must be >= low value.")
        else:
            return uniform(name, *args)


class ModelOptimizerParameters(object):
    """
    Model optimizer parameters.

    List of supported distributions:

        * *choice*: list of options to be sampled. It should be a list or a
          tuple.
        * *lognormal*: returns a value drawn from a lognormal distribution. It
          requires the mean and the standard deviation.
        * *normal*: returns a value drawn from a lognormal distribution. It
          requires the mean and the standard deviation.
        * *randint*: returns a positive random integer. It requires a maximum
          integer value.
        * *uniform*: returns a value drawn from a uniform distribution. It
          requires a minimum and maximum value.

    Example
    -------
    >>> from grmlab.modelling.model_optimization import ModelOptimizerParameters
    >>> parameters = ModelOptimizerParameters()
    >>> parameters.add("param_1", "normal", 0, 1)
    >>> parameters.add("param_2", "uniform", 2, 5)
    >>> print(parameters)
    param_1 : ('normal', 0, 1)
    param_2 : ('uniform', 2, 5)
    """
    def __init__(self):
        self._dict_params = {}
        self._parameters = {}

    def add(self, name, distribution, *args):
        """
        Add parameter to optimize.

        Parameters
        ----------
        name : str
            The name of the parameter to optimize.

        distribution : str
            The seach space distribution of a parameter.

        args : arguments
            Arguments to pass to a distribution.
        """
        self._dict_params[name] = (distribution, *args)
        self._parameters[name] = _add_distribution(name, distribution, *args)

    def _check_param(self, name):
        if name in self._dict_params.keys():
            print("Parameter {} with setting ({}) will be overwritten.".format(
                name, self._dict_params[name]))

    def __str__(self):
        return "\n".join("{} : {}".format(key, value) for key, value
                         in self._dict_params.items())
