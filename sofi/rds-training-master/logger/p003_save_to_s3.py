import mdsutils
import pandas as pd
import numpy as np
import timeit
import logging
import gc
import datetime
from smart_open import open
from logger import make_logger
import argparse


def window(file, path, create_dt):
    l, _=make_logger(log_name='Save to S3')
    bucket=['b1', 'b31', 'b61', 'b91']   
    for i in range(len(file)):
        l.info('Bucket: %s' %bucket[i])
        df=pd.read_csv(path+file[i]+'.csv')
        l.info('Total number of row: %s' %(len(df)))
        df['create_dt']=create_dt
        f = open(f's3://sofi-data-science/gen2_collection_model/{file[i]}.csv','w') 
        df.to_csv(f, index = False)
        f.close()
 
    



def main():
    l, _=make_logger(log_name='Save to S3')
    
    parser = argparse.ArgumentParser(description='save sample to s3')
    parser.add_argument('--DATA_PATH')
    args = parser.parse_args()  
    
    #for key, val in args._get_kwargs():
    #    print(key, '-->', val)

    data_dir=args.DATA_PATH
    l.info('Data location: %s' %data_dir)
    x = datetime.datetime.now()
    ps_date=x.strftime('%Y%m%d')
    l.info('Create date: %s' %ps_date)
    

    file_m=['b_1_m', 'b_31_m', 'b_61_m', 'b_91_m']
    window(file_m, data_dir, ps_date)

    file_3d=['b_1_3d', 'b_31_3d', 'b_61_3d', 'b_91_3d']
    window(file_3d, data_dir, ps_date)
    
    l.info('Save to S3 complete')



if __name__ == "__main__":
    main()