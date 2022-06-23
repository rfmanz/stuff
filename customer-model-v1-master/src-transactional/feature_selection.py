"""
Select features for each model.
"""
import lightgbm as lgb

from data import load_dataframe
from rdsutils.boruta import Boruta


src_base = os.path.dirname(os.path.realpath(__file__))
config_file_path = os.path.abspath(os.path.join(src_base, 
                                                '../config.json'))
with open(config_file_path, 'r') as f:
    CONFIG_FILE = json.load(f)


# there HAS to be a better way to do this...
# technical debt I guess...
features = ['fico_score', 'vantage_score', 'all7120', 'all8220', 'bcc2800', 
            'bcc7120', 'bcx3423', 'iln5520', 'iqt9413', 'iqt9415', 'mtf5820',
            'stu5031', 'credit_card_loan_amount', 'delinquencies_90_days', 
            'education_loan_amount', 'mortgage_loan_amount', 
            'secured_loan_amount', 'total_outstanding_balance', 
            'total_tradelines_open', 'unsecured_loan_amount', 'giact_nr_pass',
            'giact_nr_decline', 'giact_nr_other', 'fraud_score_1', 
            'fraud_score_2', 'address_risk_score', 'email_risk_score', 
            'phone_risk_score', 'name_address_correlation', 
            'name_email_correlation', 'name_phone_correlation', 
            'nr_social_profiles_found', 'days_since_first_deposit', 
            'age_money_account', 'lag_acc_open_first_transaction', 
            'first_deposit_amount', 'giact_time_since_first_link', 
            'giact_time_since_last_link', 'transaction_as_pct_of_balance', 
            'time_since_last_transaction', 'nr_past_ach_returns', 
            'nr_ach_returns_3d', 'nr_ach_returns_10d', 'nr_ach_returns_30d',
            'nr_past_check_returns', 'nr_check_returns_3d', 
            'nr_check_returns_10d', 'nr_check_returns_30d', 
            'transaction_amount', 'account_ending_balance', 'card_present_ind',
            'nr_past_deposits', 'nr_deposits_3d', 'nr_deposits_10d', 
            'nr_deposits_30d', 'nr_past_transactions', 'nr_transactions_3d', 
            'nr_transactions_10d', 'nr_transactions_30d', 
            'pct_returned_deposits', 'pct_returned_deposits_30d', 
            'dollar_val_returns', 'dollar_val_returns_3d', 
            'dollar_val_returns_10d', 'dollar_val_returns_30d',
            'mean_account_balance_3d', 'mean_account_balance_10d', 
            'mean_account_balance_30d', 'std_account_balance_3d', 
            'std_account_balance_10d', 'std_account_balance_30d',
            'sum_deposits_3d', 'sum_deposits_10d', 'sum_deposits_30d',
            'sum_withdrawals_3d', 'sum_withdrawals_10d', 'sum_withdrawals_30d', 
            'mean_deposits_3d','max_deposits_3d', 'mean_deposits_10d', 
            'mean_deposits_30d', 'mean_withdrawals_3d', 'mean_withdrawals_10d',
            'mean_withdrawals_30d', 'max_deposits_10d', 'max_deposits_30d', 
            'max_withdrawals_3d', 'max_withdrawals_10d', 'max_withdrawals_30d',
            'nr_direct_deposits', 'transaction_code_encoded']

target_col_name = 'target_1'

boruta_args = {drop_at: -15, max_iter: 50, random_state: 10, thresh: 0.3, verbose: 1}

lgb_default_params = {boosting_type: 'gbdt', 
                      metric: 'auc',
                      max_depth: 4, 
                      n_estimators: 250, 
                      colsample_bytree: 0.6, 
                      learning_rate: 0.1, 
                      reg_alpha: 10, 
                      scale_pos_weight: pos_wgt_scaling_factor, 
                      min_data_in_leaf: 50, 
                      random_state: 222}


def select_features(X, y, feature_selector):
    """ 
    X - numpy array
    y - numpy array
    """
    fsel.fit(X, y)
    
    dimp = pd.DataFrame({'feature': features, 
                         'score': fsel.scores, 
                         'mean_imp': np.mean(fsel.imps, axis=0)})\
                         .sort_values(by=['score', 'mean_imp'], 
                                      ascending=False)
    
    return dimp

    
def main():
    """ """
    target_col_name = CONFIG_FILE['target_column']
    
    df = load_dataframe('labeled', )prefix, name, base_path='data'
    
    X = 
    
    pos_wgt_scaling_factor = ~y.sum() / y.sum()
    
    clf = lgb.LGBMClassifier(boosting_type='gbdt', metric='auc', max_depth=4, n_estimators=250, colsample_bytree=0.6, 
                             learning_rate=0.1, reg_alpha=10, scale_pos_weight=pos_wgt_scaling_factor, min_data_in_leaf=50, random_state=222)
    
    fsel = Boruta(clf, drop_at=-15, max_iter=50, random_state=10, thresh=0.3, verbose=1)

    
if __name__ == '__main__':
    main()
