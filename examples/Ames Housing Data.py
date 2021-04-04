from pyutils import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax



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
full_df.drop('Id', axis=1, inplace=True)


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

skew_features = full_df[full_df.select_dtypes(['int64', 'float64']).columns].apply(lambda x: skew(x)).sort_values(
    ascending=False)
high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index
print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
for i in skew_index:
    full_df[i] = boxcox1p(full_df[i], boxcox_normmax(full_df[i] + 1))

full_df = correlated(full_df, 0.8, True)

# CV
num_bins = int(np.floor(1 + np.log2(len(y))))
sale_price_bins = pd.qcut(y, q=num_bins, labels=list(range(num_bins)))

x_train, x_test, y_train, y_test = train_test_split(
    full_df.iloc[:1460],
    y,
    random_state=12,
    stratify=sale_price_bins)

# LGBM

params = {'boosting_type': 'gbdt',
          'objective': "regression",
          'metric': 'rmse',
          'max_bin': 200,
          'max_depth': 5,
          'num_leaves': 200,
          'learning_rate': 0.01,
          'feature_fraction': 0.7,
          'bagging_fraction': 0.8,
          'bagging_freq': 4,
          'verbose': 1,
          'num_threads': 1,
          'lambda_l2': 3,
          'min_gain_to_split': 0,
          'min_sum_hessian_in_leaf': 11
          }
#sale_price_bins.iloc[y_train.index]
num_bins = 12
def train_lgbm(x_train, x_test, y_train, params, n_folds=12, early_stopping_rounds=100, num_boost_round=7000):
    global lgb_model
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
train_lgbm(full_df.iloc[:1460], full_df.iloc[1461:,], y, params)

#-------------------------------------------------
def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)

 X = full_df.iloc[:1460]
def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)

from sklearn.model_selection import KFold
kf = KFold(n_splits=12, random_state=42, shuffle=True)
train_labels = y
from sklearn.model_selection import cross_val_score

from lightgbm import LGBMRegressor
lightgbm = LGBMRegressor(objective='regression',
                       num_leaves=6,
                       learning_rate=0.01,
                       n_estimators=7000,
                       max_bin=200,
                       bagging_fraction=0.8,
                       bagging_freq=4,
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=42)

cv_rmse(lightgbm)

score = cv_rmse(lightgbm)
print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))

lgb_model_full_data = lightgbm.fit(X, train_labels)

predictions = lgb_model_full_data.predict(full_df.iloc[1460:,],num_iteration=lightgbm.best_iteration_)

#0.12107 - When submitted
sample_submission.iloc[:,1] = np.floor(np.expm1(predictions))
sample_submission.to_csv("~/Downloads/sample_submission3.csv",index=False)
