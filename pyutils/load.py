import pandas as pd
import datatable as dt
import zipfile
import re


def read_data(path_ending_with_filename=None, return_df=False, method=None):
    """
    Read single csv file or in zip.
    Available methods:
        'dt' = Datatable fread
    TODO: Add to read methods. i.e., parquet, pickle, arrow, etc.
    """
    dt.options.progress.enabled = True
    if isinstance(path_ending_with_filename, str):
        if path_ending_with_filename.endswith('.zip'):
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
                print(str(",".join(keys)))
        else:
            # SINGLE FILE
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

    else:
        # LIST OF FILES
        dfs = {}
        for x in path_ending_with_filename:
            if x.endswith('.csv'):
                if method == 'dt':
                    dfs["{0}".format(re.findall("\w+(?=\.)", x)[0])] = dt.fread(x).to_pandas()
                else:
                    dfs["{0}".format(re.findall("\w+(?=\.)", x)[0])] = pd.read_csv(x)
        keys = list(dfs.keys())
        values = list(dfs.values())
        if return_df:
            return dfs.values()
        else:
            for i in enumerate(dfs):
                print(i[1], " ","="," " ,"(",f"{values[i[0]].shape[0]:,}",":",f"{values[i[0]].shape[1]:,}",")" ,sep="")


            print(str(",".join(keys)))
