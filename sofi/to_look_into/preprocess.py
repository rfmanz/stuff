# data wrangling packages
import json
import os
#import sys
#import time
import pandas as pd
import numpy as np
import pickle
import joblib
import datetime as dt


from io import StringIO

pd.options.mode.chained_assignment = None


MODEL_DIR = os.path.join(os.getcwd(), 'model')

model_path = os.path.join(MODEL_DIR, 'b1_lightGBM_final.pkl')
ml_model = joblib.load(model_path)
lgbm=ml_model.booster_.feature_name()


# Need to make sure that the order of the inputs in the MLP list matches the dev

mlp=['Ever_DPD',
 'Num_DPD_3m_d',
 'sofidti_refresh',
 'roll_back',
 'vantage_diff_ratio',
 'repay_term_rmn_scaled',
 'premier_v1_2_all4028',
 'vantage_v3_score',
 'cycle_past_due_amount_start',
 'Ever_DPD30_3m',
 'premier_v1_2_all8152',
 'sofi_ubti',
 'premier_v1_2_all5935',
 'premier_v1_2_bca8370',
 'premier_v1_2_bcx7110',
 'premier_v1_2_brc1300',
 'past_due_to_income',
 'premier_v1_2_all8155',
 'Ever_DPD30_1m',
 'reverse_payment',
 'premier_v1_2_all5042',
 'premier_v1_2_brc3425',
 'premier_v1_2_all4001',
 'premier_v1_2_rtr7160']
     
lstm=['vantage_v3',
 'max_dpd',
 'dpd1',
 'dpd30',
 'dpd60',
 'dpd90',
 'ach_active']


no_missing=['repay_term_rmn_scaled',
 'Ever_DPD',
 'roll_back',
 'Num_DPD_3m_d',
 'cycle_past_due_amount_start',
 'Ever_DPD30_3m',
 'sofi_ubti',
 'past_due_to_income',
 'Ever_DPD30_1m',
 'reverse_payment']



with open(MODEL_DIR+'/default_value.pickle', 'rb') as f:
    default_value = pickle.load(f)


f = open(MODEL_DIR+'/premier_attr_mapper.json')
premier_attr_mapper = json.load(f)
    
    
f = open(MODEL_DIR+'/LSTM_impute_missing.json')
lstm_impute_missing = json.load(f)


f = open(MODEL_DIR+'/mlp_impute_missing.json')
mlp_impute_missing = json.load(f)


with open(MODEL_DIR+'/MLP_trans.pickle', 'rb') as f:
    mlp_transformer = pickle.load(f)

with open(MODEL_DIR+'/LSTM_trans.pickle', 'rb') as f:
    lstm_transformer = pickle.load(f)
    
    

def input_data(script_dir, rel_sql_file,  ctx):  
    ''' '''
    assert exists(script_dir)
    abs_sql_file = os.path.join(script_dir, rel_sql_file)

    with open(abs_sql_file) as sql:        
        query = sql.read()
        df= pd.read_sql(query,ctx)
        return df

def pull_date():
    model_dir = os.path.dirname(FILE) #<-- absolute dir the script is in
    sql_dir=os.path.join(model_dir, 'sql_file')
    df=input_data(sql_dir, 'b1.sql', ctx)
    df_lstm=input_data(sql_dir, 'b1_lstm.sql', ctx)
    return df, df_lstm


def features_generation(df):
    
    df['BUCKET_SCORING_DATE']=df['BUCKET_SCORING_DATE'].astype(str).str.slice(stop=10).apply(lambda x: 
                                        dt.datetime.strptime(x,'%Y-%m-%d'))
    df['DATE_FUND']=df['DATE_FUND'].astype(str).str.slice(stop=10).apply(lambda x: 
                                        dt.datetime.strptime(x,'%Y-%m-%d'))
    df['MOB']=((df.BUCKET_SCORING_DATE - df.DATE_FUND)/np.timedelta64(1, 'M')).astype(int)
    df['REPAY_TERM_RMN_SCALED']=df.REPAY_TERM_RMN/(df.INITIAL_TERM*12)
    
    df['NUM_DPD_D_SCALE']=df.NUM_DPD_D/df.MOB
    df['NUM_DPD30_D_SCALE']=df.NUM_DPD30_D/df.MOB
    df['NUM_DPD30_60_SCALE']=df.NUM_DPD30_60/df.MOB
    df['NUM_DPD1_30_6M_SCALE']=df.NUM_DPD1_30_6M/df.MOB
    df['NUM_DPD30_6M_D_SCALE']=df.NUM_DPD30_6M_D/df.MOB
    df['NUM_DPD60_D_SCALE']=df.NUM_DPD60_D/df.MOB
    
    df['MOB_PERCENT']=df.MOB/(df.INITIAL_TERM*12)
    
    df['ALL5820']=np.where(df.ALL5820>999999990,0, df.ALL5820)
    df['MTF5820']=np.where(df.MTF5820>999999990,0, df.MTF5820)
    df['ALX5839']=np.where(df.ALX5839>999999990,0, df.ALX5839)
    df['MTX5839']=np.where(df.MTX5839>999999990,0, df.MTX5839)

    df['SOFIDTI_REFRESH']=np.where(df['IS_COBORROWER']==0, (df['ALL5820']-df['MTF5820'])*12/df['GROSS_INCOME'], 
                                   (df['ALX5839']-df['MTX5839'])*12/df['GROSS_INCOME'])
    
    df['VANTAGE_DIFF_RATIO']=np.where((df.VANTAGE_V3_SCORE>=300)&(df.VANTAGE_V3_SCORE<=850)&(df.VANTAGE_SCORE_ORIG>=300)&(df.VANTAGE_SCORE_ORIG<=850),
                                      (df.VANTAGE_SCORE_ORIG-df.VANTAGE_V3_SCORE)/df.VANTAGE_SCORE_ORIG, np.nan)
    
    df['VANTAGE_DIFF']=np.where((df.VANTAGE_V3_SCORE>=300)&(df.VANTAGE_V3_SCORE<=850)&(df.VANTAGE_SCORE_ORIG>=300)&(df.VANTAGE_SCORE_ORIG<=850),
                                      (df.VANTAGE_SCORE_ORIG-df.VANTAGE_V3_SCORE), np.nan)
    

    
    return df


def mlp_encode_defaults(df, default_value):
    """Replace default values with NaN, int encode them"""
    default_encoded_cols = []
    df['exclusion_reason'] = np.nan
    for k, (v, encode) in default_value.items():
        if encode=='premier_attr':
            cname = 'premier_v1_2_'+k.lower()
        else: 
            cname = k.lower() #-> lowercase features are encoded (uppercase features are no treatment features)
        
        if encode!='lstm_dpd_feature':
            try:
                if isinstance(v, pd.Interval):
                    is_default = ~df[k].between(v.left, v.right) & ~df[k].isna()
                elif isinstance(v, list):
                    is_default = df[k].isin(k)
                else:
                    raise RuntimeError('Data type {} not supported'.format(str(type(v))))

                if ~is_default.isna().all():
                    default_encoded_cols.append(cname)
                    df[cname] = df[k]
                    df.loc[is_default, cname] = np.nan #set default values to NaN
                if ~is_default.isna().all() and encode=='dpd_feature':
                    df.loc[is_default, 'exclusion_reason'] = 'invalid input value'

            except:
                print("{} not in df ".format(k))
                
    for i in mlp:
        if i.lower() in df.columns.tolist():
            df[i]=df[i.lower()].copy()
        elif i.upper() in df.columns.tolist():
            df[i]=df[i.upper()].copy() 
    for i in no_missing:
        df['exclusion_reason']=np.where(df[i.upper()].isna()==1, 'missing information', df['exclusion_reason'])

    return df

def lgbm_decode_defaults(df, lgbm_feat_list):
    """For the LightGBM models, as the algorithm model has the ability to handle missing values and outliers, 
    no imputation is done to missing values and outliers ( including Experian error codes). """
     

    
    for k, v in premier_attr_mapper.items():
        if k in lgbm_feat_list:
            df[k]=df[v].copy()
    
    for i in lgbm_feat_list:
        if i.upper() in df.columns.tolist():
            df[i]=df[i.upper()].copy() 


    return df


def lstm_encode_defaults(df_lstm, df, default_value):
    """For the LSTM models, if the time window covers periods before the origination, the inputs are imputed by -1 (this part is done in sql ).
    Other than that, Experian error codes (vantage score=1 or 4) are replaced with missing values and imputed based on the methods as described above.
    """
    
    
    default_encoded_cols = []
    df_lstm['exclusion_reason'] = np.nan
    for k, (v, encode) in default_value.items():
        if encode=='lstm_dpd_feature' or encode=='vantage_score':
            cname = k.lower()
            try:
                if isinstance(v, pd.Interval):
                    is_default1 = ~df_lstm[k].between(v.left, v.right) & ~df_lstm[k].isna()
                    is_default2 = df_lstm[k]==-1
                    is_default=~is_default1==is_default2
                elif isinstance(v, list):
                    is_default = df_lstm[k].isin(k)
                else:
                    raise RuntimeError('Data type {} not supported'.format(str(type(v))))

                if ~is_default.isna().all():
                    default_encoded_cols.append(cname)
                    df_lstm[cname] = df_lstm[k]
                    df_lstm.loc[is_default, cname] = np.nan #set default values to NaN
                if  encode=='lstm_dpd_feature':
                    df_lstm.loc[is_default, 'exclusion'] =1

            except:
                print("{} not in df ".format(k))

                
    df_lstm['vantage_v3']=df_lstm['vantage_v3_score'].copy()
    lstm_dedup=pd.DataFrame(df_lstm.groupby(by=['LOAN_ID'])['exclusion'].sum()).reset_index(drop=False)

    df['lstm_exclusion_reason']=np.where(lstm_dedup['exclusion']==1, 'invalid input value', np.nan)

    return df_lstm, df





def impute_missing(df, feature, impute_missing_json):
    impute_vals = {}
    for f in feature:
        try:
            val = impute_missing_json[f]
            if val != None:
                impute_vals[f] = val
                df[f].fillna(val, inplace=True)
        except:
            print("{} not in df ".format(f))
    return df   


 

def normalization(df, features, transformer):
    features_xf=[]   
    for f in features:
        df[f"{f}_xf"] = transformer[f].transform(df[f].values.reshape(-1,1))
        features_xf=features_xf+[f"{f}_xf"]

    return df, features_xf



def lstm_reshape(df,features ):
    df_reshape=df[features].to_numpy().reshape(-1, 24,len(features))
    return df_reshape


def excl_score(exclusion_reason,lstm_exclusion_reason, b1_score):
    if exclusion_reason=='invalid input value' or lstm_exclusion_reason=='invalid input value':
        return 99
    elif exclusion_reason=='missing information':
        return 98
    elif exclusion_reason=='deceased':
        return 97
    else:
        return b1_score
    
    

def lstm_input_reshape(df):
    mystring=['VANTAGE_V3_SCORE',
 'MAX_DPD',
 'DPD1',
 'DPD30',
 'DPD60',
 'DPD90',
 'ACH_ACTIVE',
 'PAYMENT_AMT',
 'REVERSE_AMT']
    
    df_lstm=df[['LOAN_ID', 'BUCKET_SCORING_DATE']]
    df_lstm['SEQUENCE']=0

    for i in range(1, 24):
        temp=df[['LOAN_ID']]
        temp['SEQUENCE']=i
        df_lstm=df_lstm.append(temp)

    for j in mystring:
        name_list=[]
        string=[str(x) for x in range(0, 24)]
        name_list=name_list+[j+'_'+s for s in string]
        keep_list=['LOAN_ID']+ name_list
        df_flat=df[keep_list]
        for k in name_list:
            df=df.drop(columns=[k])
        df_wtl= pd.wide_to_long(df_flat, stubnames=j+'_', i='LOAN_ID', j='SEQUENCE')
        df_lstm=df_lstm.merge(df_wtl,left_on=['LOAN_ID', 'SEQUENCE'],right_on=['LOAN_ID', 'SEQUENCE'], how='left' )
        


    df_lstm.columns=['LOAN_ID', 'BUCKET_SCORING_DATE','SEQUENCE']+mystring
    df_lstm=df_lstm.sort_values(by=['LOAN_ID','BUCKET_SCORING_DATE', 'SEQUENCE']).reset_index(drop=True)
    #df_lstm=df_lstm.drop(columns=['SEQUENCE'])
    
    df.columns= df.columns.str.upper()
    df_lstm.columns= df_lstm.columns.str.upper()
    
    return  df, df_lstm




def lstm_flat(df_lstm, lstm_features_xf):

    key_list=['LOAN_ID','BUCKET_SCORING_DATE', 'SEQUENCE' ]
    
    string=[str(x) for x in range(0, 24)]
    df_lstm=df_lstm.sort_values(by=key_list)
 

    data=df_lstm[key_list]

    data=data[data['SEQUENCE']==0].reset_index(drop=True)
    data=data.drop(columns=['SEQUENCE'])
    lstm_features_xf_flat=[]
    for i in lstm_features_xf:   
        temp=df_lstm[[i]].to_numpy()
        trainX = temp.reshape( -1, 24)
        name_list=[i+'_'+s for s in string]
        lstm_features_xf_flat=lstm_features_xf_flat+name_list
        temp2 = pd.DataFrame(trainX , columns = name_list)
        data=data.join(temp2)
    data['BUCKET_SCORING_DATE']=data['BUCKET_SCORING_DATE'].astype(str).str.slice(stop=10).apply(lambda x: 
                                        dt.datetime.strptime(x,'%Y-%m-%d'))

    return data, lstm_features_xf_flat

def preprocess(df):
    df, df_lstm=lstm_input_reshape(df)
      
    df=features_generation(df)
    # MLP 
    df=mlp_encode_defaults(df,default_value)
    df=impute_missing(df, mlp, mlp_impute_missing)
    df, mlp_features_xf=normalization(df, mlp, mlp_transformer)
    # LGBM 
    df=lgbm_decode_defaults(df,lgbm)
    # LSTM 
    df_lstm,df =lstm_encode_defaults(df_lstm,df, default_value)
    df_lstm=impute_missing(df_lstm, lstm, lstm_impute_missing)
    df_lstm, lstm_features_xf=normalization(df_lstm, lstm, lstm_transformer)
    df_lstm_flat, lstm_features_xf_flat=lstm_flat(df_lstm, lstm_features_xf)
    
    df_lstm=lstm_reshape(df_lstm, lstm_features_xf)
    
    return df,df_lstm,df_lstm_flat, mlp_features_xf, lstm_features_xf_flat