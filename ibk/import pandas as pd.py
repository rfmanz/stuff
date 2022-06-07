import pandas as pd
import numpy as np
import json 
import os

os.getcwd()

param_Can = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})

# class SetPath():
#     def __init__(self, data):
        
#         self.hyperparams = {}
        
#         hyperparams = {
#              "param_Can": data
#         }           
#         self.hyperparams["CAN"] = hyperparams


# test = SetPath(data=param_Can)
# test.hyperparams

# np.where(param_Can['a']>1,10,param_Can['b'])
# np.where(param_Can['a']>1,param_Can['b'],10) # where this condition is met, for this column, impute with 10 

# param_Can['a'][0] = np.where(param_Can['a'][0],np.nan,None) 


# np.where(param_Can['a'],1, 0)



df = pd.read_csv('D:/Downloads/pl_test_data.csv')
df2 = np.where(df['invest_account'], df['cash_equity_v2'].fillna(0), np.nan)

with open('D:/Downloads/PL_target_encoding.json') as json_file:
    label_enc_dict = json.load(json_file)

# map target encoding to data

cols = ['employment_status', 'applicant_type']


data = df.reindex(columns = list(df) + [col+'_target_enc' for col in cols])


for col in cols:
    data[col+'_target_enc'] = data[col].map(eval(label_enc_dict[col]))

label_enc_dict[cols[0]]
label_enc_dict['consolidated_channel']

data[['target_pred_int', 'target_pred_auprc', 'target_pred_mlp']]

list(data)
df[cols]
data[cols + [col+'_target_enc' for col in cols]]

pd.options.display.max_columns = None

eval(label_enc_dict[col])


param_Can['a'].map({1:'a', 2:'b', 3:'c'})
import numpy as np
np.array([range(i, i + 3) for i in [2, 4, 6]])


x1 = np.random.randint(10, size=6) 
x1
x2 = np.random.randint(10, size=(3, 4))
x2
x3 = np.random.randint(10, size=(3, 4, 5))
x3


p = (1,2)
e = (1,2)
 
np.array(p)
np.array(e)

b = np.array([[[1, 2, 3],[4, 5, 6]],[[7, 8, 9],[10, 11, 12]]])
b[0][1][0]




