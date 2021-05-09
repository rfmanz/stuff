import pandas as pd
import datatable as dt
import zipfile
import re
import os

def directory(directory_path):
    """Puts you in the right directory. Gives you list of files in path"""
    os.chdir(re.findall("^(.*[\\\/])", directory_path)[0])
    csv_files = os.listdir(directory_path)
    return csv_files

def get_csv_names_from_list(paths):
    if not isinstance(paths, list):
        raise TypeError('We need a list of csv file paths here')
    dfs = []
    for i in paths:
        if i.endswith('.csv'):
            df_name = re.findall("\w+(?=\.)", i)[0]
            dfs.append(df_name)
    print(str(",".join(dfs)))
    print(str(".shape,".join(dfs)), ".shape", sep='')

def read_data(path_ending_with_filename=None, return_df=False, method=None):
    """
    Reads single csv or list of csvs or csvs in zip.

    Available methods:
        'dt' = Datatable fread

    TODO: Add to read methods. i.e., parquet, pickle, arrow, etc.
    """
    dt.options.progress.enabled = True
    if isinstance(path_ending_with_filename, str):
        if path_ending_with_filename.endswith('.zip'):
            zf = zipfile.ZipFile(path_ending_with_filename)
            files = zf.namelist()
            if return_df:
                dfs = {}
                for x in files:
                    if x.endswith('.csv'):
                        if method == 'dt':
                            dfs["{0}".format(re.findall("\w+(?=\.)", x)[0])] = dt.fread(zf.open(x)).to_pandas()
                        else:
                            dfs["{0}".format(re.findall("\w+(?=\.)", x)[0])] = pd.read_csv(zf.open(x))
                keys = list(dfs.keys())
                values = list(dfs.values())
                for i in enumerate(dfs):
                    print(i[1], ":", values[i[0]].shape)
                return dfs.values()
            else:
                filelist = zf.filelist
                csv_file_names = [format(re.findall("\w+(?=\.)", zf.namelist()[i])[0]) for i in
                                  range(len(zf.namelist())) if zf.namelist()[i].endswith('.csv')]
                file_pos = [i for i, x in enumerate(zf.namelist()) if x.endswith('.csv')]
                uncompressed = [f"{(zf.filelist[i].file_size / 1024 ** 2):.2f} Mb" for i in file_pos]
                compressed = [f"{(zf.filelist[i].compress_size / 1024 ** 2):.2f} Mb" for i in file_pos]

                print(pd.concat([pd.Series(csv_file_names), pd.Series(uncompressed), pd.Series(compressed)], axis=1,
                                keys=["file_names", "uncompressed", "compressed"]))
                print()
                print(*csv_file_names, sep=",")


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
        # LIST OF CSV FILES
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
                print(i[1], " ", "=", " ", "(", f"{values[i[0]].shape[0]:,}", ":", f"{values[i[0]].shape[1]:,}", ")",
                      sep="")

            print(str(",".join(keys)))



