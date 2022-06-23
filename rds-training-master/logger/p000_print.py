import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import datetime as dt

import json
import os
from os.path import abspath, expanduser, join, basename, exists, dirname 
import sys
import time

import pandas as pd
import psycopg2
import csv
import argparse
import numpy as np
from pandas.tseries.offsets import MonthEnd
import os
import glob
import logging

def get_epoch():
    return str(int(dt.datetime.now().timestamp()))

def get_tmp_file():
    return 'tmp/' + get_epoch() + '.csv'


def get_time(start_time):
    return time.time() - start_time

def append(rows, row):
    if row:
        rows.append(row)
        





def input_data(script_dir, rel_sql_file, condition, connection): 
    # start timer
    
    print('Starting Query')

    start_time = time.time()
    count = 0
    rows = []
    try:
        # read sql query from file
        assert exists(script_dir)
        
        abs_sql_file = os.path.join(script_dir, rel_sql_file)
        with open(abs_sql_file) as sql:
            query = sql.read().split('${CONDITION}')
            copy = ''
            for i in range(len(query)):
                copy = copy + query[i]
                if i < len(condition):
                    copy = copy + condition[i]
                    #if type(condition[i]) is list:
                     #   copy = copy + "','".join([str(j) for j in condition[i]])
                    #else:
                        #copy = copy + condition[i]
            file = ''
            os.makedirs('tmp/', exist_ok=True)
            tmp = os.listdir('tmp/')
            if tmp and tmp[len(tmp) - 1] != '.DS_Store':
                file = 'tmp/' + tmp[len(tmp) - 1]
                print('Existing TSV file %s being reused %.4fs' % (file, get_time(start_time)))
            else:
                # process records using server-side cursor
                file = get_tmp_file()
                with connection.cursor() as cursor:
                    with open(file, 'wb') as tmp:
                        print('Executing copy statement for records in %.4fs' % get_time(start_time))
                        cursor.copy_expert(copy, tmp)
                        print('Records copied from Postgres in %.4fs' % get_time(start_time))
                        
            with open(file, 'r') as item:
                total = sum(1 for _ in item)
                print('Processing %s records from file in %.4fs' % (str(total), get_time(start_time)))
                
                
            with open(file, 'r') as tmp:
                print('Reading TSV from Postgres records in %.4fs' % get_time(start_time))
                csv.field_size_limit(sys.maxsize)
                tsv = csv.DictReader(tmp, delimiter='\t')
                print('Records being parsed from TSV in %.4fs' % get_time(start_time))

                timer = time.time()
                for record in tsv:
                    row=record
                    append(rows, row)
                df = pd.DataFrame().from_records(rows)
                print('total record=', len(df))
                
                
                
                # df.to_csv(get_csv_file())

            # print time to process records
            print('%s records in %.4fs' % (count, get_time(start_time)))
            os.remove(file)
            return df

    except:
        print('Error: %s records in %.4fs' % (count, get_time(start_time)))
        # pd.DataFrame().from_records(rows).to_csv(get_csv_file())
        raise
        
        
def input_data_1(script_dir, rel_sql_file, condition, conn):  

    assert exists(script_dir)
    abs_sql_file = os.path.join(script_dir, rel_sql_file)

    with open(abs_sql_file) as sql:        
        query = sql.read().split('${CONDITION}')
        copy = ''
        for i in range(len(query)):
            copy = copy + query[i]
            if i < len(condition):
                copy = copy + condition[i]
        df= pd.read_sql(copy,conn)
        #conn.close()
        #engine.dispose()

        return df

def main():
    
    parser = argparse.ArgumentParser(description='get target profile')
    parser.add_argument('--SQL_PATH')
    args = parser.parse_args()  
    
    for key, val in args._get_kwargs():
        print(key, '-->', val)
        
    sql_dir=args.SQL_PATH
    
    
    conn = psycopg2.connect(
    database='dw-prod-sofidw-ro',
    user='psung',
    password='sofi',
    host='localhost',
    port='15501'
    )

    condition=['2020-02-03', '2020-02-03']
 
    profile=input_data(sql_dir, 'applications_file.sql' ,condition, conn)

 
    profile.to_csv('example.csv', index=False)
    print('Done!!!')


    
        
if __name__ == "__main__":
    main()