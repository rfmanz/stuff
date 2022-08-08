"""
GRMlab base classes.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from inspect import signature
from sklearn.base import BaseEstimator


def _print_class(class_name, dict_params):
    """
    Print GRMlab class `params`.

    Parameters
    ----------
    class_name : str
        The name of the class.

    dict_params: dict
        The dictionary of paramaters.
    """
    n_params = len(dict_params)
    line_header = "{}(\n".format(class_name)
    line_param = "{}={},\n"
    line_param_last = "{}={})"
    lines = [line_header]

    for i, (key, value) in enumerate(sorted(dict_params.items())):
        str_value = str(value) if type(value) is float else repr(value)
        if i < n_params-1:
            lines.append(line_param.format(key, str_value))
        else:
            lines.append(line_param_last.format(key, str_value))
    str_lines = "    ".join(lines)
    return str_lines


class GRMlabBase(BaseEstimator):
    """
    Base class for all GRMlab classes.

    Notes
    -----
    All classes should specify all the parameters explicitly in ``__init__``,
    i.e., keyword arguments (no ``*args`` or ```**kwargs``). This class is
    inspired by `sklearn.base.BaseEstimator`.
    """
    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the class."""

        # Inspect the class constructor arguments to find parameters
        if cls.__init__ is object.__init__:
            # No explicit constructor to introspect
            return []

        init_signature = signature(cls.__init__)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != "self" and p.kind != p.VAR_KEYWORD]

        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "GRMlab classes must specify their "
                    "parameters in the signature of their __init__. "
                    "{} with constructor {} does not follow this "
                    "convention.".format(cls, init_signature))

        # Extract and sort argument names excluding self
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get the parameters for this class.

        Parameters
        ----------
        deep : boolean (default=True)
            If True, return the parameters for this class and contained
            subojects that are inherited from GRMlabBase.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        dict_params = {}
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, "get_params"):
                # extract params from inner classes inherited from GRMlabBase.
                deep_items = value.get_params().items()
                dict_params.update(("{}__{}".format(key, k), val)
                                   for k, val in deep_items)
            dict_params[key] = value
        return dict_params

    def set_params(self, **params):
        """
        Set the parameters of this class.

        The method works on simple classes as well as on nested objects (nested
        classes). The latter have parameters of the form
        ``<component>__<parameter>``, therefore elements of nested classes can
        be updated.
        """
        if not params:
            return self

        class_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)
        for key, value in params.items():
            key, delimiter, sub_key = key.partition("__")
            if key not in class_params:
                raise ValueError("Invalid parameter {} for class {}. "
                                 "Check the list of valiable parameters with "
                                 "`class.get_params().keys()`."
                                 .format(key, self))
            elif delimiter:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                class_params[key] = value

        for key, sub_params in nested_params.items():
            class_params[key].set_params(**sub_params)

        return self

    def __repr__(self):
        return _print_class(self.__class__.__name__, self.get_params(
            deep=False))

    def __getstate__(self):
        try:
            state = super(GRMlabBase, self).__getstate__()
        except AttributeError:
            state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        try:
            super(GRMlabBase, self).__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)


class GRMlabVariable(GRMlabBase):
    """"""
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype


class GRMlabProcess(GRMlabBase, metaclass=ABCMeta):
    """"""
    @abstractmethod
    def run(self):
        raise NotImplementedError

    @abstractmethod
    def transform(self):
        raise NotImplementedError
