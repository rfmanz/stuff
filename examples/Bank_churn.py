#https://www.kaggle.com/c/customer-churn-prediction-2020
from pyutils import * 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

path  = '/home/r/Downloads/customer-churn-prediction-2020.zip'
read_data(path)
sampleSubmission,test,train = read_data(path,True)
sampleSubmission.shape,test.shape,train.shape
peek(train,3)
peek(test,3)

#eda

# I need a function that gives a fact grid graph thing to see the distributions of all variables
# by type and see if they the distributions are the same for the train and test. 

#fe {

test_id = test.id

ad= pd.concat([train.drop("churn",axis=1), test.drop("id",axis=1)],axis=0)
ad.shape

def correlated(df, threshold, drop_columns=False, encode_type='dumy'):
    '''Create a copy if you're viewing before deleting.
    If deleting df= correlated(df,...)'''

    if bool((df.select_dtypes('object')).size > 0):
        df2 = encode(df, encode_type)
        df_corr = df2.corr()

    else:
        df_corr = df.corr()

    triangle = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))
    to_drop = pd.Series(df_corr.iloc[:,
                        np.where((df_corr.mask(np.tril(np.ones(df_corr.shape, dtype=bool))).abs() > threshold).any())[
                            0]].columns)

    if drop_columns:
        dftodrop = df.drop(labels=to_drop, axis=1)
        return dftodrop


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

ad2 = correlated(ad,0.8,True,encode_type='lbl')
ad2.shape
peek(ad2,3)
peek(ad,3)
#}


# This competition is about predicting whether a customer will change telecommunications provider, something known as "churning".
# 
# The training dataset contains 4250 samples. Each sample contains 19 features and 1 boolean variable "churn" which indicates the class of the sample. The 19 input features and 1 target variable are:
# 
# "state", string. 2-letter code of the US state of customer residence
# "account_length", numerical. Number of months the customer has been with the current telco provider
# "area_code", string="area_code_AAA" where AAA = 3 digit area code.
# "international_plan", . The customer has international plan.
# "voice_mail_plan", . The customer has voice mail plan.
# "number_vmail_messages", numerical. Number of voice-mail messages.
# "total_day_minutes", numerical. Total minutes of day calls.
# "total_day_calls", numerical. Total minutes of day calls.
# "total_day_charge", numerical. Total charge of day calls.
# "total_eve_minutes", numerical. Total minutes of evening calls.
# "total_eve_calls", numerical. Total number of evening calls.
# "total_eve_charge", numerical. Total charge of evening calls.
# "total_night_minutes", numerical. Total minutes of night calls.
# "total_night_calls", numerical. Total number of night calls.
# "total_night_charge", numerical. Total charge of night calls.
# "total_intl_minutes", numerical. Total minutes of international calls.
# "total_intl_calls", numerical. Total number of international calls.
# "total_intl_charge", numerical. Total charge of international calls
# "number_customer_service_calls", numerical. Number of calls to customer service
# "churn", . Customer churn - target variable
