import pandas as pd
import datatable as dt
import zipfile
import re


def read_data(path_ending_with_filename=None, zip_path=None, method=None):
    """ Creates df with the names of the files. Either single file 'path_ending_with_filename' or 'zip_path' for zip files.
    pd.read_csv is default. Method option ='dt' for reading with datatable. TODO: expand read methods

    do this with a dictionary:
    zip_path='/home/r/Downloads/house-prices-advanced-regression-techniques.zip'
    zf = zipfile.ZipFile(zip_path)
    files = zf.namelist()
    d={}
    for x in files[1:]:
    d["{0}".format(x)] = pd.read_csv(zf.open(x))
    train = d['train.csv']

    """
    if path_ending_with_filename:
        fl = path_ending_with_filename.lower()
    else:
        fl = zip_path.lower()

    csv_files = []
    if fl.endswith('.zip'):
        print('unzipping...')
        zf = zipfile.ZipFile(zip_path)
        files = zf.namelist()
        print(files)
        for i in files:
            if i.endswith('.csv'):
                to_read = i.split('.')[0]
                csv_files.append(to_read)
        for i in csv_files:
            if method == 'dt':
                globals()[f"{i}"] = dt.fread(zf.open(i + '.csv'))
                globals()[f"{i}"] = globals()[f"{i}"].to_pandas()

            else:
                globals()[f"{i}"] = pd.read_csv(zf.open(i + '.csv'))
                print("Variable created:", f"{i}")
    else:
        df_name = re.findall("\w+(?=\.)", path_ending_with_filename)[0]
        if method == 'dt':
            globals()[f"{df_name}"] = dt.fread(path_ending_with_filename)
            globals()[f"{df_name}"] = globals()[f"{df_name}"].to_pands()
        else:
            globals()[f"{df_name}"] = pd.read_csv(path_ending_with_filename)
        print("Variable created:", f"{df_name}")
