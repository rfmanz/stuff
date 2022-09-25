# manifold mixup!

alpha = 0.2
batch_size = 1000
_ = plt.hist(np.random.beta(alpha, alpha, size=batch_size), bins=100)

# selection round 2
fts_ = ['p13_upl8132', 'p13_bcc8322', 'p13_bcc5520',  # 't11_taua02q7', 
        'p13_bcn5020', 't11_tall3205', 'p13_upl0438',  # 'p13_aut0416',
        'p13_all7312', 'p13_all7518', 'p13_rta7300', 'p13_iqz9420', 
        'p13_iqf9540', 'p13_iqt9510', 'p13_iqz9425', 't11_trev0722', 
        't11_tstu2752', 't11_tiln2755', 't11_tstu1755', 't11_tmti2752', # - might be too good for fl issue
        't11_tiln2754', 'p13_all7936', 'p13_all8352', 'p13_cru1300',
        'p13_reh7120', 't11_tbca2526', 't11_tbca3530', 't11_tbca4504',
        't11_tbca2381', 't11_tbcc3305',
        't11_tbcc1303', 'p13_upl8320', 'p13_aut8140',
        'p13_cru8320', 'p13_alm6160',
        'p13_alj8120', 'p13_rtr5520',
       ]

# remove 'p13_aua8811', 'p13_iln0316'

fts_ = list(set(fts_))
mc = [monotone_dict[f] for f in fts_]

params = lgbm_bmk.get_params().copy()
display(len(fts_), len(mc))
params["monotone_constraints"] = mc
params["scale_pos_weight"] = None
params["early_stopping_rounds"] = 50
params["verbose_eval"] = -1


# lgbm_ = lgb.LGBMClassifier(**params)
# lgbm_.fit(train_df[fts_],
#           train_df[target], 
#           train_df[weight],
#           eval_set=(valid_df[fts_], valid_df[target]),
#           eval_sample_weight=[valid_df["weight_eval"]],
#           verbose=False
#           )

# models[f"lgbm_{len(fts_)}"] = lgbm_

train_data = lgb.Dataset(train_df[fts_], train_df[target], feature_name=fts_, weight=train_df[weight])
valid_data = lgb.Dataset(valid_df[fts_], valid_df[target], feature_name=fts_, weight=valid_df["weight_eval"])
test_data = lgb.Dataset(test_df[fts_], test_df[target], feature_name=fts_, weight=test_df["weight_eval"])


# This one works!
# DEFINE CUSTOM LOSS FUNCTION
def logloss(preds, data):
    y_true = data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    weight = data.get_weight() if data.get_weight() is not None else 1
    grad = (preds - y_true) * weight
    hess = preds * (1.0 - preds) * weight
    return grad, hess

# DEFINE CUSTOM EVAL LOSS FUNCTION
def logloss_eval(preds, data):
    y_true = data.get_label()
    weight = data.get_weight() if data.get_weight() is not None else np.ones(len(y_true))
    preds = 1. / (1. + np.exp(-preds))
    sum_loss = sum((-(y_true * np.log(preds)) - ((1 - y_true) * np.log(1 - preds))) * weight)
    
    return 'binary_logloss', sum_loss / sum(weight), False


from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder


n_classes = 2

def logregobj(preds, train_data):
    labels = train_data.get_label().astype(int)
    labels = np.eye(n_classes)[labels]
    
    preds = np.reshape(preds, (len(labels), n_classes))
    preds = np.exp(preds)
    preds = np.multiply(preds, 1/np.sum(preds, axis=1)[:, np.newaxis])
    weight = train_data.get_weight() if train_data.get_weight() is not None else 1
    weight = weight.reshape(-1, 1)
    
    grad = (preds - labels) * weight
    hess = 2.0 * preds * (1.0-preds) * weight
    return grad.flatten("F"), hess.flatten("F")


def logregobj_eval(preds, data):
    y_true = data.get_label()
    weight = data.get_weight() if data.get_weight() is not None else np.ones(len(y_true))
    preds = np.reshape(preds, (len(labels), n_classes))
    preds = np.exp(preds)
    preds = np.multiply(preds, 1/np.sum(preds, axis=1)[:, np.newaxis])
    preds = preds[:, 1]  # since we are doing bce
    sum_loss = sum((-(y_true * np.log(preds)) - ((1 - y_true) * np.log(1 - preds))) * weight)
    
    return 'binary_logloss', sum_loss / sum(weight), False

n_classes = 2
alpha = 0.2

def logregobj_mm(preds, train_data):
    labels = train_data.get_label().astype(int)
    labels = np.eye(n_classes)[labels]
    n = len(labels)
    
    preds = np.reshape(preds, (n, n_classes))
    preds = np.exp(preds)
    preds = np.multiply(preds, 1/np.sum(preds, axis=1)[:, np.newaxis])
    weight = train_data.get_weight() if train_data.get_weight() is not None else 1
    weight = weight.reshape(-1, 1)
    
    # fun part! introducing MM
    idx = list(range(n))
    np.random.shuffle(idx)
    preds_mix = preds[idx, :].copy()
    labels_mix = labels[idx, :].copy()
    lam = np.random.beta(alpha, alpha, size=(n,1))
    preds = np.multiply(preds, lam) + np.multiply(preds_mix, 1-lam)
    labels = np.multiply(labels, lam) + np.multiply(labels_mix, 1-lam)
    
    
    grad = (preds - labels) * weight
    hess = 2.0 * preds * (1.0-preds) * weight
    return grad.flatten("F"), hess.flatten("F")


def logregobj_mm_eval(preds, data):
    labels = data.get_label()
    weight = data.get_weight() if data.get_weight() is not None else np.ones(len(y_true))
    preds = np.reshape(preds, (len(labels), n_classes))
    preds = np.exp(preds)
    preds = np.multiply(preds, 1/np.sum(preds, axis=1)[:, np.newaxis])
    preds = preds[:, 1]  # since we are doing bce
    sum_loss = sum((-(labels * np.log(preds)) - ((1 - labels) * np.log(1 - preds))) * weight)
    
    return 'binary_logloss', sum_loss / sum(weight), False


lgbm_params = {
    'objective': 'binary',
    'random_seed': 0
    }
%time model = lgb.train(lgbm_params, train_data)
y_pred = model.predict(valid_df[fts_])
y_test = valid_df[target]
print(roc_auc_score(y_test, y_pred))
# >> 0.5492623974996498

# # custome fn v1
# lgbm_params = {
#     'random_seed': 0
#     }
# model = lgb.train(lgbm_params, 
#                   train_data,
#                   fobj=logloss,
#                   feval=logloss_eval)
# # Note: When using custom objective, the model outputs logits
# y_pred = model.predict(valid_df[fts_])
# y_test = valid_df[target]
# print(roc_auc_score(y_test, y_pred))
# # >> 0.5491950102587455

# custom fn v2
lgbm_params = {
'random_seed': 0,
'objective': "multiclass",
'num_class': 2
}
%time model = lgb.train(lgbm_params, train_data, fobj=logregobj, feval=logloss_eval)
# Note: When using custom objective, the model outputs logits
y_pred = model.predict(valid_df[fts_])[:,1]
y_test = valid_df[target]
print(roc_auc_score(y_test, y_pred))
# >> 0.5448764109297742

# custom fn v2
lgbm_params = {
'random_seed': 0,
'objective': "multiclass",
'num_class': 2
}
%time model = lgb.train(lgbm_params, train_data, fobj=logregobj_mm, feval=logregobj_mm_eval)
# Note: When using custom objective, the model outputs logits
y_pred = model.predict(valid_df[fts_])[:,1]
y_test = valid_df[target]
print(roc_auc_score(y_test, y_pred))
# >> 0.5448764109297742

custom_params = params.copy()
custom_params["objective"] = "multiclass"
custom_params["num_class"] = 2
custom_params["metric"] = "auc_mu"


%%time 
model = lgb.train(custom_params, 
                  train_data, 
                  valid_sets=[valid_data],
                  fobj=logregobj_mm, 
#                   feval=logregobj_mm_eval,
                 )


mname = f"lgbm_{len(fts_)}_mm"
models[mname] = model
valid_df[f"pred_{mname}"] = model.predict(valid_df[model.feature_name()])[:,1]
test_df[f"pred_{mname}"] = model.predict(test_df[model.feature_name()])[:,1]


pred_cols = ["pred_gen3", "pred_lgbm_35", "pred_lgbm_37", "pred_lgbm_35_mm"]
metrics = get_pred_reports(test_df, target, pred_cols, sample_weight_col="weight_eval")
metrics["% inc in ks"] = metrics["ks"] / metrics.loc["pred_gen3", "ks"]
metrics.sort_values("ks", inplace=True)
metrics



for ris in ["booked", "proxy", "others"]:
    print(f"===================== {ris} =====================")
    df_ = test_df[test_df.ri_source == ris]
    metrics = get_pred_reports(df_, target, pred_cols, sample_weight_col="weight_eval")
    metrics["% inc in ks"] = metrics["ks"] / metrics.loc["pred_gen3", "ks"]
    metrics.sort_values("ks", inplace=True)
    metrics
    display(metrics)
    
    
np.average(df_unimp[target], weights=df_unimp[weight])