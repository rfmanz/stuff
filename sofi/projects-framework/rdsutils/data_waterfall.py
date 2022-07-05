import numpy as np
import pandas as pd
import gc


class DataWaterfallReport:
    
    def __init__(self, df, function_sequence, 
                 col_to_expand=None, index_formatter=None, 
                 column_formatter=None):
        """
        generate data waterfall for binary classification task
        
        Caution: col_to_expand does not automatically deal with None/Nan. 
        Encode to string for better performance
        
        @params df: dataframe 
                    - df of starting population
        @params function_sequence: list of functions
                    - each function takes in single argument: df
                      and returns a single df with remaining rows for next round
                    - Note: function name will be used as report index
        @params col_to_expand: str 
                    - categorical columns to expend into columns
                    - with each column recording a count for each categorical value
        @params inplace: bool
                    - df will be modified if inplace=True, will save memory
        @params index_formatter: function
                    - string formatting function for each index string
        @params colum_formatter: function
                    - string formatting function for each column string
        """
        
        self.df = df
        self.functions = function_sequence
        self.col_to_expand = col_to_expand
        if col_to_expand is not None:
            col_order = df[col_to_expand].value_counts()
            col_order = col_order.index.to_list()
        else:
            col_order = []

        # reporting data
        # dictionary: 
        # key - function
        # values - (#excluded, #remaining, #good, #bad)
        self.record_index = []
        self.record_cols = ['Excluded', 'Remain'] + col_order
        self.record = []  
        self.waterfall = None
        
        # formatting
        if index_formatter is None:
            self.index_formatter = self.default_index_formatter
        else:
            self.index_formatter = index_formatter
            
        if column_formatter is None:
            self.column_formatter = self.default_column_formatter
        else:
            self.column_formatter = column_formatter   
                
    
    def default_column_formatter(self, col_name):
        return f'# {col_name}'.title()
    
    
    def default_index_formatter(self, index):
        return ' '.join(index.split('_')).title()
    
    
    def format_report(self, records, index, columns, 
                      index_formatter, column_formatter, present_ready):
        
        record_index = [index_formatter(idx) for idx in index]
        record_cols = [column_formatter(col) for col in columns]
        
        if present_ready:
            # convert every cell to string format
            records_ = []
            
            
            for row in range(len(records)):
                record_ = []
                for col in range(len(records[0])):
                    elem = records[row][col]
                    record_.append('' if pd.isna(elem) else str(int(elem)))
                records_.append(record_)
                
            records = records_
            
        result = pd.DataFrame(records, 
                              index=record_index,
                              columns=record_cols)
            
        return result
    
    
    def get_report(self, present_ready=False):
        """
        @params present_ready: bool
            - convert all cells to string
        """
        
        df = self.df.copy()
        
        col2xp = self.col_to_expand
        
        # starter row
        row = []
        n_excluded = None
        n_remain = len(df)
        row.extend([n_excluded, n_remain])
        
        if col2xp is not None:
            row.extend(df[col2xp].value_counts().values.tolist())
        
        record_index = ["Starting Population"]
        records = [row]
        
        for fn in self.functions:
            row = []
            
            # build another row
            record_index.append(fn.__name__)
            df = fn(df)
            n_excluded = n_remain - len(df)
            n_remain = len(df)
            row.extend([n_excluded, n_remain])
            
            if col2xp is not None:
                row.extend(df[col2xp].value_counts().values.tolist())
            
            records.append(row)
        
        self.record_index = record_index
        self.records = records
        
        self.waterfall = self.format_report(records, record_index,
                                            self.record_cols,
                                            self.index_formatter,
                                            self.column_formatter,
                                            present_ready)
        
        return self.waterfall
            