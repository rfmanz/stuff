


from pyutils import * 
import pandas as pd 


read_data("D:/Downloads/credit_abuser.zip")

credit_abuser,_credit_abuser = read_data("D:/Downloads/credit_abuser.csv.zip", return_df=True, method='dt')


credit_abuser.shape
_credit_abuser.head()
peek(credit_abuser)
credit_abuser.to_csv('D:/Downloads/credit_abuser.csv', )

credit_abuser.to_clipboard()
credit_abuser.head()

pd.Series(credit_abuser.columns).to_clipboard()

describe_df(credit_abuser)

identify_missing(credit_abuser, missing_threshold=0.5)
correlated(credit_abuser, 0.8)


# high missing 
# ['ALJ5030', 'ALJ5730', 'ALL2001', 'ALL2012', 'ALL8107', 'ALL8123', 'ALL8151', 'ALL8152', 'ALL8154', 'ALL8157', 'ALL8160', 'ALL8163', 'ALL8164', 'ALL8171', 'ALL8183', 'ALL8257', 'ALL8259', 'ALL8325', 'ALL8552', 'ALL8558', 'ALL9110', 'ALL9120', 'ALL9123', 'ALL9130', 'ALL9140', 'ALL9141', 'ALL9148', 'ALL9220', 'ALL9221', 'AUA5320', 'AUA5421', 'AUA5820', 'AUA7201', 'AUT5020', 'AUT5122', 'AUT5620', 'AUT7110', 'BCA0401', 'BCA5021', 'BCA8160', 'BUS5020', 'BUS5120', 'COL2767', 'COL3238', 'COL3243', 'COL8197', 'CRU0416', 'CRU5320', 'CRU8320', 'FIP0416', 'FIP0436', 'FIP0437', 'FIP0438', 'FIP1300', 'FIP1380', 'FIP2000', 'FIP2320', 'FIP2328', 'FIP2350', 'FIP2358', 'FIP2380', 'FIP2388', 'FIP2800', 'FIP5020', 'FIP5120', 'FIP5320', 'FIP5420', 'FIP5520', 'FIP5820', 'FIP6200', 'FIP6280', 'FIP8120', 'FIP8220', 'FIP8320', 'GLBDECS', 'HLC0401', 'HLC0402', 'HLC0416', 'HLC0436', 'HLC0437', 'HLC0438', 'HLC0700', 'HLC1402', 'HLC2000', 'ILJ8120', 'ILJ8220', 'IQA9415', 'IQA9416', 'IQA9540', 
# 'IQB9415', 'IQB9416', 'IQB9417', 'IQC9410', 'IQC9415', 'IQC9416', 'IQF9410', 'IQF9415', 'IQF9416', 'IQF9510', 'IQF9540', 'IQM9415', 'IQM9416', 'IQM9540', 'IQR9415', 'IQR9416', 'IQT9412', 'IQT9413', 'IQT9415', 'IQT9416', 'IQT9533', 'IQT9535', 'IQT9536', 'MTA0400', 'MTA0416', 'MTA0436', 'MTA0437', 'MTA0700', 'MTA1370', 'MTA2800', 'MTA5020', 'MTA5030', 'MTA5400', 'MTA5830', 'MTA7433', 'MTA8120', 'MTA8150', 'MTA8153', 'MTA8157', 'MTA8160', 'MTA8370', 'MTF0416', 'MTF4260', 'MTF5020', 'MTF5129', 'MTF5820', 'MTF5838', 'MTF5930', 'MTF7110', 'MTF8111', 'MTF8120', 'MTF8132', 'MTF8140', 'MTF8166', 'MTF8222', 'MTF8320', 'MTF8810', 'MTJ0316', 'MTJ0416', 'MTJ5030', 'MTJ5320', 'MTJ5820', 'MTJ8120', 'MTS5020', 'MTS5620', 'MTX5839', 'P13_ALL2012', 'P13_ALL8107', 'P13_ALL8123', 'P13_ALL8152', 'P13_ALL8155', 'P13_ALL8163', 'P13_ALL8352', 'P13_ALL9120', 'P13_ALL9141', 'P13_ALL9220', 'P13_ALL9221', 'P13_AUA5820', 'P13_BCN8220', 'P13_COL2767', 'P13_COL3243', 'P13_ILJ0416', 'P13_ILJ5030', 'P13_ILN7150', 'P13_ILN8152', 'P13_IQB9417', 'P13_MTA5830', 'P13_MTF7110', 'P13_MTF8166', 'P13_MTJ5820', 'P13_MTJ8220', 'P13_STU0852', 'P13_STU2007', 'P13_STU2550', 'P13_STU2558', 'P13_STU5820', 'P13_STU6200', 'P13_STU6280', 'P13_STU7118', 'P13_STU8120', 'P13_STU8151', 'PIL5020', 'PIL5320', 'PIL6200', 'PIL8132', 'REJ5030', 'REJ5320', 'REV8150', 'RTI0436', 'RTI5020', 'RTI5320', 'SOFI3002', 'SOFI3006', 'SOFICLM', 'SOFIDIL', 'SOFIEDUC', 'SOFIPR36', 'SOFIPRJA', 'SOFIPRSJ', 'SOFIPRTL', 'SOFIRVLV3MO', 'SOFITRCCC', 'SOFIUNIL6', 'STU0336', 'STU0337', 'STU0436', 'STU0437', 'STU0438', 'STU1300', 'STU2550', 'STU2558', 'STU4180', 'STU5020', 'STU5031', 'STU5092', 'STU5820', 'STU6200', 'STU7110', 'STU7118', 'STU8120', 'STU8125', 'STU8132', 'STU8151', 'STU8228', 'STU8320', 'T11_TRTR3282', 'USE5030', 'USE8220', 'SOFITBKM'] 

# ['vintage', 'curret_st_bal', 'credit_line_orig']


# for experian vars (3 letters followed by 4 digits), here is the dict

target = pd.read_csv("D:/Downloads/credit_abuser_flag_202203.csv")

[col for col in  credit_abuser.columns if col[:3].isalpha() and col[4:].isdigit() or col.startswith('P13') or col.endswith('')]



pd.Series(credit_abuser.columns).isalpha()
pd.Series(credit_abuser.columns).str[:4].isalpha()

pd.Series(credit_abuser.columns).str.contains('^[A-Z]{3}[0-9]{4}$')


nimit = pd.read_clipboard()

pd.DataFrame(peek(credit_abuser)).to_clipboard()

nimit

credit_abuser.existing_products.value_counts()
credit_abuser.general_segment.value_counts()
credit_abuser.app_fruad_status.value_counts()
credit_abuser.experian_attribute


credit_abuser.existing_products.head()


# region ///1. parse existing_product fields into separate columns ///





"MONEY"
"INVEST"
"PERSONAL_LOAN"
"MORTGAGE"
"STUDENT_LOAN"
