import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import miceforest as mf


def convert_dtypes(df,varsN , varsC ):

    df[varsN] = df[varsN].astype(int)
    df[varsC] = df[varsC].astype('category')
    

    return df

def identify_missing(df, missing_threshold, drop=False):
    """Find the features with a fraction of missing values above `missing_threshold`"""

    # Calculate the fraction of missing in each column
    missing_series = df.isnull().sum() / df.shape[0]
    missing_stats = pd.DataFrame(missing_series).rename(columns={'index': 'feature', 0: 'missing_fraction'})

    # Sort with highest number of missing values on top
    missing_stats = missing_stats.sort_values('missing_fraction', ascending=False)

    # Find the columns with a missing percentage above the threshold
    record_missing = pd.DataFrame(
        missing_series[missing_series > missing_threshold]) \
        .reset_index() \
        .rename(columns={'index': 'feature', 0: 'missing_fraction'})

    to_drop = list(record_missing['feature'])

    if drop:
        df.drop(to_drop, axis=1, inplace=True)
        print(f"Columns dropped : {to_drop}")
        print(df.shape)

    else:
        print('%d features with greater than %0.2f missing values.\n' % (len(to_drop), missing_threshold))
        print(to_drop)


def encode(df, method="dmy"):
    """ methods:
        Dummy encoding = pandas get_dummies
        Label encoding = sklearn LabelEncoders
    """
    cols = df.select_dtypes('object').columns

    if method == "lbl":
        for i in cols:
            df[i] = LabelEncoder().fit_transform(df[i].astype(str))
        return df

    if method == "dmy":
        encoded = pd.get_dummies(df[cols])
        df = df.drop(columns=cols, axis=1)
        df = pd.concat([df, encoded], axis=1)
        return df


def correlated(df, threshold, drop_columns=False, encode_type='dmy'):
    '''Create a copy if you're viewing before deleting.
    If deleting df= correlated(df,...)'''
    df = df.copy()
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
        df = df.drop(labels=to_drop, axis=1)
        return df


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


def standard_scaler(df):
    """ 
    Standardization: scales features such that the distribution is centered around 0, with a standard deviation of 1.
    Normalization: shrinks the range such that the range is now between 0 and 1 (or -1 to 1 if there are negative values).
    Robust Scaler: similar to normalization but it instead uses the interquartile range, so that it is robust to outliers.

    Min-max scaling (many people call this normalization) is the simplest: values are shifted and rescaled so that they end up ranging from 0 to 1. We do this by subtracting the min value and dividing by the max minus the min. Scikit-Learn provides a transformer called MinMaxScaler for this. It has a feature_range hyperparameter that lets
    you change the range if, for some reason, you don’t want 0–1.

    Standardization is different: first it subtracts the mean value (so standardized values always have a zero mean), and then it divides by the standard deviation so that the resulting distribution has unit variance. Unlike min-max scaling, standardization does not bound values to a specific range, which may be a problem for some algorithms (e.g., neural networks often expect an input value ranging from 0 to 1). However, standardization is much less affected by outliers. For example, suppose a district had a median income equal to 100 (by mistake). Min-max scaling would then crush all the other values from 0–15 down to 0–0.15, whereas standardization would not be much affected. Scikit-Learn provides a transformer called StandardScaler for standardization. 
    """

    return pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns, index=df.index)


def mice_imputer(data):
    """https://github.com/AnotherSamWilson/miceforest"""
    na_cols = data[data.columns[data.isna().any()]]

    # Create kernel.
    kds = mf.KernelDataSet(
        na_cols,
        save_all_iterations=True,
        random_state=1991
    )
    # Run the MICE algorithm for 3 iterations
    kds.mice(3)

    # Return the completed kernel data
    completed_data = kds.complete_data()

    return completed_data


def target_encode(train, valid, col, target='target', kfold=5, smooth=20, verbose=True):
    
    """
    

    So you're extracting information from categorical features, how important is this categorical to predicting the target. 
    You're saying: for categorical variable brand, with many brands as it's categories you're grouping brand1, brand2, etc and then seeing that brand 1 has a x% probability of being purchased which is the target in this example. And then you can bundles of brands of different brands, in order to say, ok when you view brand 1 and brand2 the probability of purchase is x.  

    https://youtu.be/uROvhp7cj6Q?t=2833

    example :  
    train, test = target_encode2(train,test,['Pclass', 'Sex', 'Embarked','age_bin'], target='Survived')

        train:  train dataset
        valid:  validation dataset
        col:   column which will be encoded (in the example RESOURCE)
        target: target column which will be used to calculate the statistic


    """

    # We assume that the train dataset is shuffled
    train['kfold'] = ((train.index) % kfold)
    # We keep the original order as cudf merge will not preserve the original order
    train['org_sorting'] = np.arange(len(train), dtype="int32")
    # We create the output column, we fill with 0
    col_name = '_'.join(col) + '_' + str(smooth)
    train['TE_' + col_name] = 0.
    for i in range(kfold):
        ###################################
        # filter for out of fold
        # calculate the mean/counts per group category
        # calculate the global mean for the oof
        # calculate the smoothed TE
        # merge it to the original dataframe
        ###################################

        df_tmp = train[train['kfold'] != i]
        mn = df_tmp[target].mean()
        df_tmp = df_tmp[col + [target]].groupby(col).agg(['mean', 'count']).reset_index()
        df_tmp.columns = col + ['mean', 'count']
        df_tmp['TE_tmp'] = ((df_tmp['mean'] * df_tmp['count']) + (mn * smooth)) / (df_tmp['count'] + smooth)
        df_tmp_m = train[col + ['kfold', 'org_sorting', 'TE_' + col_name]].merge(df_tmp, how='left', left_on=col,
                                                                                right_on=col).sort_values(
            'org_sorting')
        df_tmp_m.loc[df_tmp_m['kfold'] == i, 'TE_' + col_name] = df_tmp_m.loc[df_tmp_m['kfold'] == i, 'TE_tmp']
        train['TE_' + col_name] = df_tmp_m['TE_' + col_name].fillna(mn).values

    ###################################
    # calculate the mean/counts per group for the full training dataset
    # calculate the global mean
    # calculate the smoothed TE
    # merge it to the original dataframe
    # drop all temp columns
    ###################################    

    df_tmp = train[col + [target]].groupby(col).agg(['mean', 'count']).reset_index()
    mn = train[target].mean()
    df_tmp.columns = col + ['mean', 'count']
    df_tmp['TE_tmp'] = ((df_tmp['mean'] * df_tmp['count']) + (mn * smooth)) / (df_tmp['count'] + smooth)
    valid['org_sorting'] = np.arange(len(valid), dtype="int32")
    df_tmp_m = valid[col + ['org_sorting']].merge(df_tmp, how='left', left_on=col, right_on=col).sort_values(
        'org_sorting')
    valid['TE_' + col_name] = df_tmp_m['TE_tmp'].fillna(mn).values

    valid = valid.drop('org_sorting', axis=1)
    train = train.drop('kfold', axis=1)
    train = train.drop('org_sorting', axis=1)
    return (train, valid)


def memory_usage_mb(df, *args, **kwargs):
    """Dataframe memory usage in MB. """
    return df.memory_usage(*args, **kwargs).sum()


def reduce_memory_usage(df, deep=True, verbose=True, categories=True):
    # All types that we want to change for "lighter" ones.
    # int8 and float16 are not include because we cannot reduce
    # those data types.
    # float32 is not include because float16 has too low precision.
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    start_mem = 0
    if verbose:
        start_mem = memory_usage_mb(df, deep=deep)

    for col, col_type in df.dtypes.iteritems():
        best_type = None
        if col_type == "object":
            df[col] = df[col].astype("category")
            best_type = "category"
        elif col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            df[col] = pd.to_numeric(df[col], downcast=downcast)
            best_type = df[col].dtype.name
        # Log the conversion performed.
        # if verbose and best_type is not None and best_type != str(col_type):
        # print(f"Column {col} converted from {col_type} to {best_type}")

    if verbose:
        end_mem = memory_usage_mb(df, deep=deep)
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        print(f"Memory usage decreased from"
            f" {start_mem:.2f}MB to {end_mem:.2f}MB"
            f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)")


