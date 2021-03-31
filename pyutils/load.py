import pandas as pd
import datatable as dt
import zipfile
import re


def read_data(path_ending_with_filename=None, return_df=False, zip_path=None, method=None):
    """
    Read single csv file or in zip.
    Available methods:
        'dt' = Datatable fread
    TODO: Add to read methods. i.e., parquet, pickle, arrow, etc.
    """
    if path_ending_with_filename:
        fl = path_ending_with_filename.lower()
    else:
        fl = zip_path.lower()

    if fl.endswith('.zip'):
        zip_path = '/home/r/Downloads/house-prices-advanced-regression-techniques.zip'
        zf = zipfile.ZipFile(zip_path)
        files = zf.namelist()
        dfs = {}
        for x in files:
            if x.endswith('.csv'):
                dfs["{0}".format(re.findall("\w+(?=\.)", x)[0])] = pd.read_csv(zf.open(x))
        keys = list(dfs.keys())
        values = list(dfs.values())
        if return_df:
            return dfs.values()
        else:
            for i in enumerate(dfs):
                print(i[1], ":", values[i[0]].shape)
            # print("---" * 18)
            # print("Copy to assign ↓ (.i.e train,test = read_data(...))")
            # print("---" * 18)
            print(str(",".join(keys)))
    else:
        df_name = re.findall("\w+(?=\.)", path_ending_with_filename)[0]
        if method == 'dt':
            df = dt.fread(path_ending_with_filename)
            df = df.to_pandas()
        else:
            df = pd.read_csv(path_ending_with_filename)
        if return_df:
            return df
        else:
            print(df_name, df.shape)


