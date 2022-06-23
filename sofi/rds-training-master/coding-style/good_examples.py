import pandas as pd
import dask as dd
from pandas import DataFrame
import pandas.api.types as ptypes


MAGIC_NR = 1
MAGIC_NR2 = 299


class HeyHeyHey:
    """
    class examples

    Example usage
    * design choices
    *...
    """

    def __init__(self, message: str):
        """
        doc string
        """
        self.message = message


def im_hungry(message: str) -> str:
    """
    doc string

    @params message: str
        incoming message
    @return message: str
    """
    print(1)

    # comment here
    print("you sure?")  # sounds good

    return message


a = 1 + 2


def fn(a, b):
    return a + b


def fn2(a, b):
    return fn(a + b)

