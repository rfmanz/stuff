grades = []
grades.append((95, 0.45))
grades.append((85, 0.55))
total = sum(score * weight for score, weight in grades)
total_weight = sum(weight for _, weight in grades)
average_grade = total / total_weight

from dataclasses import dataclass


@dataclass
class Position:
    name: str
    lon: float
    lat: float


Position(name, lon, lat)

Position.__delattr__(self, name)


# https://khuyentran1401.github.io/Efficient_Python_tricks_and_tools_for_data_scientists/Chapter2/dataclasses.html

from dataclasses import dataclass
from typing import List


@dataclass
class Dog:
    names: str
    age: int


@dataclass
class Dogs:
    names: List[str]
    ages: List[int]

    # def __post_init__(self):
    #     self.info = [(name, age) for name, age in zip(self.names, self.ages)]
    def __post_init__(self):
        self.info = [(name, age) for name, age in dict(zip(self.names, self.ages))]


names = ["Bim", "Pepper"]
ages = [5, 6]
dogs = Dogs(names, ages)
dogs.names
dogs.info
type(dogs.info)


class Building(object):
    def __init__(self, floors):
        self._floors = [None] * floors

    def occupy(self, floor_number, data):
        self._floors[floor_number] = data

    def get_floor_data(self, floor_number):
        return self._floors[floor_number]


building1 = Building(4)  # Construct a building with 4 floors
building1.occupy(0, "Reception")
building1.occupy(1, "ABC Corp")
building1.occupy(2, "DEF Inc")
print(building1.get_floor_data(2))

# iterrows

# https://github.com/twolodzko/getter/blob/main/getter/decorators.py


from dataclasses import dataclass


@dataclass()
class Student:
    name: str
    clss: int
    stu_id: int
    marks: []
    avg_marks: float

    def average_marks(self):
        return sum(self.marks) / len(self.marks)


student1 = Student("HTD", 10, 17, [11, 12, 14], 50.0)

print(student)

Student(name="HTD", clss=10, stu_id=17, marks=[11, 12, 14], avg_marks=50.0)

student.average_marks()


from dataclasses import dataclass, field


@dataclass()
class Student:
    name: str
    clss: int
    stu_id: int
    # marks: []
    # avg_marks: float = field(init=False)

    # def (self):
    #     self.avg_marks = sum(self.marks) / len(self.marks)


student.average_marks

student = Student("HTD", 10, 17, [98, 85, 90])
student = Student("HTD", 10, 17)

print(student)

Student(name="HTD", clss=10, stu_id=17, marks=[98, 85, 90], avg_marks=91.0)

print(student.__dataclass_fields__)

np.random.rand

x1 = np.random.randint(10, size=6)
x1
x2 = np.random.randint(10, size=(3, 4))
x2
x3 = np.random.randint(10, size=(3, 4, 5))
x3


# Pipes


def pipe(first, *args):
    for fn in args:
        first = fn(first)
    return first


from math import sqrt
from datetime import datetime


def as_date(s):
    return datetime.strptime(s, "%Y-%m-%d")


def as_character(value):
    # Do whatever as.character does
    return value


pipe("2014-01-01", as_date)
pipe(12, sqrt, lambda x: x**2, as_character)


from rich import pretty

pretty.install()
["Rich and pretty", True]

from pyutils import *

import rdsutils

from rdsutils.datasets import Dataset, StructuredDataset, DataLoader, DataDumper


# TODO:

# !!!! understand unittests
## !!! decoratoes |  attrs
# __next__, __iter__
## !! tox
# @abstractmethod
#### Data loader class
# Sql from json
# Load with different methods
# Process function

import os

os.getcwd()
from datasets.dataloader import DataLoader

DataLoader()


# pd.clip
# pd.str.slice(stop=10) # get first characters of string
# df['MOB']=((df.BUCKET_SCORING_DATE - df.DATE_FUND)/np.timedelta64(1, 'M')).astype(int)
# class DataWrangler:
# FeatureSelector


def my_sum(*args):
    result = 0
    # Iterating over the Python args tuple
    for x in args:
        result += x
    return result


print(my_sum(1, 2, 3))

import pandas as pd


pl_base = pd.read_parquet(
    "C:/Users/rfrancis/Downloads/df_final2.parquet.gzip", engine="pyarrow"
)

pl_base2 = pl_base.iloc[:, 1:50].copy()
del pl_base
import gc

hyperparams = {}

from dataclasses import dataclass


@dataclass
class SetPath:
    def __init__(self):
        self.hyperparams = {}

    def read_csv(self):
        pl_base = pd.read_parquet(
            "C:/Users/rfrancis/Downloads/df_final2.parquet.gzip", engine="pyarrow"
        )
        pl_base2 = pl_base.iloc[:, 1:50].copy()
        del pl_base
        import gc

        gc.collect()
        hyperparams = {"param_mapInst": pl_base2, "pp": pl_base2.iloc[1:50]}
        self.hyperparams["MAPS_INST"] = hyperparams

    def get_hyperparam(self):
        return self.hyperparams


p = SetPath()

p.read_csv()
p.hyperparams["MAPS_INST"]["param_mapInst"]
p.hyperparams["MAPS_INST"]["pp"].head()

person = {}
person["pets"] = {"dog": {"Fido", "asdsasd"}, "cat": "Sox"}
person

SetPath().__dataclass_fields__
self.hyperparams["MAPS_INST"] = hyperparams

gc.collect()


data_dir = "C:/Users/rfrancis/Downloads/"


print(list(pl_base2), end="")

from pathlib import Path


class DataLoader:
    """
    Purpose
    * provide a path and return a generator of dfs to be loaded
    * process the df
    * method to save the df

    files are loaded in the order of file name
    """

    def __init__(self, path, suffix="parquet", **kwargs):
        self.i = 0

        p = Path(path)
        self.files = sorted(list(p.glob(f"*.{suffix}")))
        self.kwargs = kwargs

        try:
            self.load_fn = getattr(pd, f"read_{suffix}")
        except:
            raise ValueError(
                f"trying read_{suffix}, suppported read type - must be one from pd.read_* "
            )

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):

        if self.i < len(self.files):
            fpath = self.files[self.i]
            fname = fpath.stem
            df = self.load_fn(fpath, **self.kwargs)
            self.i += 1

            return fname, df

        raise StopIteration()

    def get_paths(self):
        return list(map(lambda fpath: str(fpath.absolute()), self.files))

    def get_full(self):
        dfs = [self.load_fn(fpath, **self.kwargs) for fpath in self.files]
        return pd.concat(dfs, axis=0)


COLS = [
    "dw_application_id",
    "application_type",
    "requested_amount",
    "initial_term",
    "interest_rate",
    "fraud_type",
    "applied_city",
    "applied_state",
    "applied_zip",
    "applied_cbsa_name",
    "applied_cbsa_code",
    "employer_name",
    "loan_purpose",
    "tier",
    "credit_score",
    "fico",
    "vantage",
    "gross_income",
    "initial_decision",
    "sofi_score",
    "pl_custom_score",
    "reason",
]

dl = DataLoader(data_dir, suffix="csv")
dl.files
from IPython.display import display

dl.__dict__
df = dl.get_full()
df.shape
dl.get_paths()
dl.next()
for fname, df__ in dl:
    print(fname, df__.shape)

read_data(data_dir)

display(dl.get_paths())

import numpy as np

a = np.arange(10)
np.where(a < 5)


a = 2
b = [1, 2, 3]
c = b
a.__add__(2)
(2).__add__(3)


def outer(x):
    def inner(y):
        return x + y

    return inner


outer(15)

add15 = outer(15)
add15(10)
add15.__closure__[0].cell_contents
add15.__closure__[0].cell_contents


add15.__call__(10)
add15.__getattribute__()


from time import time
import time

time.perf_counter()
import functools


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time()
        value = func(*args, **kwargs)
        print(f"Elapsed time is {time() - start} ms")
        return value

    return wrapper


@debug
@timeit
@timer
def any_func():
    count = 0
    for number in range(10000):
        count += number
    return count


any_func()

count = 0
for number in range(10):
    count += number
    print(number, count)


for number in range(10):
    print(number)


from attrs import asdict, define, make_class, Factory
import functools


@define
class SomeClass:
    a_number: int = 42
    list_of_numbers: list[int] = Factory(list)

    def hard_math(self, another_number):
        return self.a_number + sum(self.list_of_numbers) * another_number


sc = SomeClass(1, [1, 2, 3])
sc.hard_math(3)

dir(SomeClass())
dir(outer(1)(2))
