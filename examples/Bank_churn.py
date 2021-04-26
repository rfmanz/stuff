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
ad2 = correlated(ad,0.8,True,encode_type='dmy')
y = train.churn.map(dict(yes=1,no=0))
print_split()
x_train, x_test, y_train, y_test  = split(ad2.iloc[:4250],y)






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
