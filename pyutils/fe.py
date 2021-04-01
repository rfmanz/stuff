import pandas as pd
from sklearn.preprocessing import LabelEncoder
# import miceforest
# https://github.com/AnotherSamWilson/miceforest #The-MICE-Algorithm
#
#
#
# from missingpy import MissForest
# imputer = MissForest()
# X_imputed = imputer.fit_transform(X)
# https://github.com/epsilon-machine/missingpy


def identify_missing(df, missing_threshold, drop=False):
    """Find the features with a fraction of missing values above `missing_threshold`"""

    # Calculate the fraction of missing in each column
    missing_series = df.isnull().sum() / df.shape[0]
    missing_stats = pd.DataFrame(missing_series).rename(columns={'index': 'feature', 0: 'missing_fraction'})

    # Sort with highest number of missing values on top
    missing_stats = missing_stats.sort_values('missing_fraction', ascending=False)

    # Find the columns with a missing percentage above the threshold
    record_missing = pd.DataFrame(
        missing_series[missing_series > missing_threshold])\
        .reset_index()\
        .rename(columns= {'index': 'feature',0: 'missing_fraction'})

    to_drop = list(record_missing['feature'])

    if drop:
        df.drop(to_drop, axis=1, inplace=True)
        print(f"Columns dropped : {to_drop}")
        print(df.shape)

    else:
        print('%d features with greater than %0.2f missing values.\n' % (len(to_drop), missing_threshold))
        print(to_drop)


def encode(df, method="dumy"):
    """ methods:
        Dummy encoding = pandas get_dummies
        Label encoding = sklearn LabelEncoders
    """
    cols = df.select_dtypes('object').columns

    if method == "lbl":
        for i in cols:
            df[i] = LabelEncoder().fit_transform(df[i].astype(str))
        return df

    if method == "dumy":
        encoded = pd.get_dummies(df[cols])
        df = df.drop(columns=cols, axis=1)
        df = pd.concat([df, encoded], axis=1)
        return df


def correlated(df, threshold, drop_columns=False, encode_type='dumy'):
    '''Create a copy if you're viewing before deleting.
    If deleting df= correlated(df,...)'''

    if bool((df.select_dtypes('object')).size > 0):
        df = encode(df, encode_type)
        df_corr = df.corr()


    else:

        df_corr = df.corr()

    triangle = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))
    to_drop = pd.Series(df_corr.iloc[:,
                        np.where((df_corr.mask(np.tril(np.ones(df_corr.shape, dtype=bool))).abs() > threshold).any())[
                            0]].columns)

    if drop_columns:
        df.drop(labels=to_drop, axis=1, inplace=True)
        return df
        # print("Columns dropped:")
        # print(to_drop)
        # print(df.shape)

    else:

        collinear = pd.DataFrame(columns=['drop_feature', 'corr_feature', 'corr_value'])

        for i in to_drop:

            # Find the correlated features
            corr_features = list(triangle.index[triangle[i].abs() > threshold])

            # Find the correlated values
            corr_values = list(triangle[i][triangle[i].abs() > threshold])
            drop_features = [i for _ in range(len(corr_features))]

            # Record the information (need a temp df for now)
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                              'corr_feature': corr_features,
                                              'corr_value': corr_values})
            # Add to dataframe
            collinear = collinear.append(temp_df, ignore_index=True)



        else:
            print(f"From {len(df.columns)} columns")
            print(f"{len(to_drop)} highly correlated columns to drop:")
            print()
            print(to_drop)
            print("-----")
            print(
                "Note_to_self...drop_feature may be duplicated due to multiple, stronger than threshold, correlated pairs:")
            print()
            print(collinear)
