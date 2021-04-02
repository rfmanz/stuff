from pyutils import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

# Load

# read_data("/home/r/Downloads/house-prices-advanced-regression-techniques.zip")
sample_submission, test, train = read_data("/home/r/Downloads/house-prices-advanced-regression-techniques.zip", True)
# sample_submission.shape,test.shape,train.shape


# EDA
# describe_df(full_df)

# FE
y = train.SalePrice
y = np.log(y)
full_df = pd.concat([train.drop(columns='SalePrice'), test])


# identify_missing(full_df,0.8,True)
# https://www.kaggle.com/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition/notebook#Feature-Engineering:
def handle_missing(features):
    # the data description states that NA refers to typical ('Typ') values
    features['Functional'] = features['Functional'].fillna('Typ')
    # Replace the missing values in each of the columns below with their mode
    features['Electrical'] = features['Electrical'].fillna("SBrkr")
    features['KitchenQual'] = features['KitchenQual'].fillna("TA")
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

    # the data description stats that NA refers to "No Pool"

    # features["PoolQC"] = features["PoolQC"].fillna("None")
    # Replacing the missing values with 0, since no garage = no cars in garage
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        features[col] = features[col].fillna(0)
    # Replacing the missing values with None
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')
    # NaN values for these categorical basement features, means there's no basement
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None');

    # Group the by neighborhoods, and fill in missing value by the median LotFrontage of the neighborhood
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    objects = []
    for i in features.columns:
        if features[i].dtype == object:
            objects.append(i)
    features.update(features[objects].fillna('None'))

    # And we do the same thing for numerical features, but this time with 0s
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numeric.append(i)
    features.update(features[numeric].fillna(0))
    return features


handle_missing(full_df)
full_df = correlated(full_df, 0.8, True)


full_df.drop('Id', axis=1, inplace=True)

# CV
num_bins = int(np.floor(1 + np.log2(len(y))))
sale_price_bins = pd.qcut(y, q=num_bins, labels=list(range(num_bins)))

x_train, x_test, y_train, y_test = train_test_split(
    full_df.iloc[:1460],
    y,
    random_state=12,
    stratify=sale_price_bins

)

params = {'boosting_type': 'gbdt',
          'objective': "regression",
          'metric': 'rmse',
          'max_bin': 300,
          'max_depth': 5,
          'num_leaves': 200,
          'learning_rate': 0.01,
          'feature_fraction': 0.7,
          'bagging_fraction': 0.7,
          'bagging_freq': 10,
          'verbose': 0,
          'num_threads': 1,
          'lambda_l2': 3,
          'min_gain_to_split': 0,
          }

y_train.iloc[sale_price_bins]
sale_price_bins.iloc[y_train.index]


def train_lgbm(x_train, x_test, y_train, params, n_folds=5, early_stopping_rounds=100, num_boost_round=1500):
    training_start_time = time()

    features = list(x_train.columns)
    oof = np.zeros(len(x_train))
    predictions = np.zeros(len(y_train))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(x_train, sale_price_bins.iloc[y_train.index])):
        print(f'Training fold {fold + 1}')

        train_set = lgb.Dataset(x_train.iloc[trn_idx], label=y_train.iloc[trn_idx])
        val_set = lgb.Dataset(x_train.iloc[val_idx], label=y_train.iloc[val_idx])

        lgb_model = lgb.train(params,
                              train_set=train_set,
                              valid_sets=[train_set, val_set],
                              num_boost_round=num_boost_round,
                              early_stopping_rounds=early_stopping_rounds)

        oof[val_idx] = lgb_model.predict(x_train.iloc[val_idx][features],
                                         num_iteration=lgb_model.best_iteration)

        # predictions += lgb_model.predict(x_test[features])/n_folds
        print('-' * 50)
        print('\n')

    oof_rmse = np.sqrt(metrics.mean_squared_error(y_train, oof))
    print(f"Total training time: {str(datetime.timedelta(seconds=time() - training_start_time)).split('.')[0]}")
    print(f'RMSE: is {oof_rmse}')
    return lgb_model


train_lgbm(x_train, x_test, y_train, params)
