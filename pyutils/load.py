import pandas as pd
import datatable as dt
import zipfile
import re
import os
import time
from datetime import timedelta
import sys


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


def read_data(path_ending_with_filename=None, return_df=False, method=None, dataframes=None):
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

            if dataframes:

                dataframes = [x.strip(" ") for x in dataframes.split(",")]

                if len(dataframes) == 1:
                    x = dataframes[0] + '.csv'
                    dfs = {}
                    if method == 'dt':
                        dfs["{0}".format(re.findall("\w+(?=\.)", x)[0])] = dt.fread(zf.open(x)).to_pandas()
                    else:
                        dfs["{0}".format(re.findall("\w+(?=\.)", x)[0])] = pd.read_csv(zf.open(x))

                    keys = list(dfs.keys())
                    values = list(dfs.values())
                    for i, k in enumerate(dfs):
                        print(i + 1, ".", " ", k, " ", "=", " ", "(", f"{values[i].shape[0]:,}", " ", ":", " ",
                              f"{values[i].shape[1]:,}", ")",
                              sep="")
                    if return_df:
                        return pd.DataFrame.from_dict(values[0])
                else:
                    files = [x + '.csv' for x in dataframes]
            else:
                files = zf.namelist()

            if return_df:
                dfs = {}
                start_time = time.monotonic()
                for x in files:
                    if x.endswith('.csv'):
                        if method == 'dt':
                            dfs["{0}".format(re.findall("\w+(?=\.)", x)[0])] = dt.fread(zf.open(x)).to_pandas()
                        else:
                            dfs["{0}".format(re.findall("\w+(?=\.)", x)[0])] = pd.read_csv(zf.open(x))
                end_time = time.monotonic()
                print(timedelta(seconds=end_time - start_time))

                keys = list(dfs.keys())
                values = list(dfs.values())
                for i, k in enumerate(dfs):
                    print(i + 1, ".", " ", k, " ", "=", " ", "(", f"{values[i].shape[0]:,}", " ", ":", " ",
                          f"{values[i].shape[1]:,}", ")",
                          sep="")
                return dfs.values()
            else:
                if not dataframes:
                    csv_file_names = [format(re.findall("\w+(?=\.)", zf.namelist()[i])[0]) for i in
                                      range(len(zf.namelist())) if zf.namelist()[i].endswith('.csv')]
                    # if dataframes:
                    #
                    #     file_pos = [i for i, x in enumerate(csv_file_names)]

                    # else:
                    file_pos = [i for i, x in enumerate(zf.namelist()) if x.endswith('.csv')]

                    uncompressed_dir = [f"{(zf.filelist[i].file_size / 1024 ** 2):.2f} Mb" for i in file_pos]
                    compressed = [f"{(zf.filelist[i].compress_size / 1024 ** 2):.2f} Mb" for i in file_pos]

                    print(pd.concat([pd.Series(csv_file_names), pd.Series(uncompressed_dir), pd.Series(compressed)], axis=1,
                                    keys=["file_names", "uncompressed", "compressed"]))
                    print()
                    print(*csv_file_names, sep=",")
        else:
            # SINGLE FILE
            if path_ending_with_filename.endswith(".csv"):
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
                # CSVS IN DIRECTORY
                dfs = {}
                os.chdir(path_ending_with_filename)
                if dataframes:
                    dataframes = [x.strip(" ") for x in dataframes.split(",")]
                    csvs_in_directory = [x for x in os.listdir(path_ending_with_filename) if x.endswith('.csv')]
                    files = list(set(csvs_in_directory) & set([x + '.csv' for x in dataframes]))
                else:
                    files = [x for x in os.listdir(path_ending_with_filename) if x.endswith('.csv')]
                for x in files:
                    if method == 'dt':
                        dfs["{0}".format(re.findall("\w+(?=\.)", x)[0])] = dt.fread(x).to_pandas()
                    else:
                        dfs["{0}".format(re.findall("\w+(?=\.)", x)[0])] = pd.read_csv(x)
                keys = list(dfs.keys())
                values = list(dfs.values())
                if return_df:
                    for i, k in enumerate(dfs):
                        print(i + 1, ".", " ", k, " ", "=", " ", "(", f"{values[i].shape[0]:,}", " ", ":", " ",
                              f"{values[i].shape[1]:,}", ")",
                              sep="")

                    return dfs.values()
                else:

                    uncompressed_dir = [f"{(sys.getsizeof(dfs[i]) / 1024 ** 2):.2f} Mb" for i in dfs]

                    print(pd.concat([pd.Series(keys), pd.Series(uncompressed_dir)], axis=1,
                                    keys=["file_names", "uncompressed"]))
                    print()
                    print(*keys, sep=",")

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
            for i, k in enumerate(dfs):
                print(i + 1, ".", " ", k, " ", "=", " ", "(", f"{values[i].shape[0]:,}", " ", ":", " ",
                      f"{values[i].shape[1]:,}", ")",
                      sep="")
