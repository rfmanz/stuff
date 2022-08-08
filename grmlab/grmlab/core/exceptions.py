"""
GRMlab custom exceptions.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.


class NotExecutedException(Exception):
    """
    Exception class to raise if the dependent process has not been executed.
    This class inherits from Exception base class.

    Parameters
    ----------
    arg : object
        The class object.

    msg : str
        The exception message.
    """
    def __init__(self, arg, msg):
        self.arg = arg
        self.msg = msg

        super().__init__(msg.format(arg.__class__.__name__))


class NotFittedException(NotExecutedException):
    """
    Exception class to raise if instance is used before fitting.

    Parameters
    ----------
    arg : object
        The class object.
    """
    def __init__(self, arg):
        msg = "This {} instance is not fitted yet. Run fit() first."
        super().__init__(arg, msg)


class NotSolvedException(NotExecutedException):
    """
    Exception class to raise if instance is used before solving.

    Parameters
    ----------
    arg : object
        The class object.
    """
    def __init__(self, arg):
        msg = "This {} instance is not solved yet. Run solve() first."
        super().__init__(arg, msg)


class NotRunException(NotExecutedException):
    """
    Exception class to raise if class method step is used before running.

    Parameters
    ----------
    arg : object
        The class object.

    step : str or None (default=None)
        The class method step name.
    """
    def __init__(self, arg, step=None):
        if step is None:
            super().__init__(arg, "This {} instance has not been run. "
                             "Run first.")
        else:
            super().__init__(arg, "This {0} instance has not completed step "
                             "{1}. Run {1}() first.".format(arg, step))


class NotRunAnyMethodException(NotExecutedException):
    """
    Exception class to raise if instance is used before any of its available
    methods is run.

    Parameters
    ----------
    arg : object
        The class object.
    """
    def __init__(self, arg, methods):
        msg = "Run any of the available methods from ({}).".format(
            ",".join(methods))
        msg = "This {} instance has not run any method. " + msg
        super().__init__(arg, msg)
