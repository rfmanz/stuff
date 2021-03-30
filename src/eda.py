import pandas as pd
import numpy as np
import sys
from tabulate import tabulate

def describe_df(df):
#Numerical
    print("--" * 20)
    print('Columns:', df.shape[1])
    print('Rows:', df.shape[0])
    print("Memory usage:", (f"({(sys.getsizeof(df) / 1024 ** 2):.2f} Mb)"))


    print("--"*20)
    print('NUMERICAL VARIABLES:')

    numerical = df.select_dtypes(include=np.number)
    concatenated_numerical = pd.concat([
        round(numerical.isnull().sum() / df.shape[0] * 100, 2).astype(str) + "%",
        numerical.isnull().sum(),
        numerical.count(),
        numerical.min(),
        numerical.mean(),
        numerical.max()
    ], axis=1, keys=["%NULLS", "COUNT_NULLS", "NOT_NULL", 'MIN', 'MEAN', 'MAX'], sort=False).sort_values('COUNT_NULLS', ascending=False).reset_index().rename(columns={'index': ''})

    t = numerical.mode().T
    t.rename(columns={0: 'MODE'}, inplace=True)
    concatenated_numerical = concatenated_numerical.merge(t, how='left', left_on='', right_on=t.index)
    concatenated_numerical.index = concatenated_numerical.index + 1
    concatenated_numerical = concatenated_numerical.iloc[:, [0, 4, 5, 6, 7, 1, 2, 3]]

    print(tabulate(concatenated_numerical, headers=
    [
        'MIN',
        'MEAN',
        'MAX',
        'MODE',
        "%NULLS",
        "#_NULLS",
        "NOT_NULL",

    ], tablefmt="presto", colalign=("right"), floatfmt='.3f'))

# Categorical

    print('-----'*20)
    print()
    print('CATEGORICAL VARIABLES:')
    categorical = df.select_dtypes('object')
    if categorical.shape[1] == 0:
        print("No Categorical Variables")
    else:
        concatenated_categorical = pd.concat([

            round(categorical.isnull().sum() / df.shape[0] * 100, 2).astype(str) + "%",

            categorical.isnull().sum(),
            categorical.count()
        ],

            keys=["%NULLS",
                  "COUNT_NULLS",
                  "NOT_NULL"

                  ], axis=1, sort=False).sort_values('%NULLS', ascending=False).reset_index().rename(
            columns={'index': ''})

        max_unique = 5
        u_strs = []

        for col in categorical:
            series = categorical.loc[categorical[col].notnull(), col]
            n_unique = series.nunique()
            if n_unique > max_unique:
                u_strs.append(str(n_unique) + ' unique values')
            else:
                u_strs.append(str(series.unique()))

        t = pd.DataFrame(u_strs, categorical.columns)
        t = t.reset_index()
        t = t.rename(columns={'index': '', 0: 'Unique_Values'})
        concatenated_categorical = concatenated_categorical.merge(t, on='')
        concatenated_categorical.index = concatenated_categorical.index + 1

        print(tabulate(concatenated_categorical, headers=
        [
            "%NULLS",
            "#_NULLS",
            "NOT_NULL",
            "Unique_Values"

        ], tablefmt="presto",colalign=("left")))





