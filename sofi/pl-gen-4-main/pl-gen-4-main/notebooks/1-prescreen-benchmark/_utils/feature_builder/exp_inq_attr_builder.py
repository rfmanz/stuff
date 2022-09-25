import pandas as pd

class exp_inq_attr_builder:
    
    params_0 = {'descriptor': 'num_inq',
                'source_table': 'inquiry',
                'loan_type': '',
                'window': '',
                'window_unit':'',
                'sublist':""}
    
    def __init__(self, param=params_0):
        self.param = param
    
    def _get_attr_name_loan_type(self, loan_type, if_long_description = 0):
        if loan_type != '':
            if if_long_description ==0:
                return '_' + loan_type
            else:
                return loan_type.replace('_', " ")
        else:
            return ''

    def _get_attr_name_sublist(self,sublist, if_long_description = 0):
        if sublist != '':
            if if_long_description ==0:
                return '_list' 
            else:
                return " lender in list " + sublist
        else:
            return ''

    def _get_attr_name_window(self,window, window_unit, if_long_description = 0):
        if (window != '') & (window_unit != ''):
            if if_long_description ==0:
                return '_' + window + window_unit
            else:
                return " in last " + window + " " + window_unit
        else:
            return ''

    def get_attr_name(self, if_long_description = 0):
        params_inq = self.param
        descriptor = params_inq['descriptor']
        source_table = params_inq['source_table']
        try:
            loan_type=params_inq['loan_type']
        except:
             loan_type=''
        try:
            window=params_inq['window']
        except:
             window=''
        try:
            window_unit=params_inq['window_unit']
        except:
             window_unit=''
        try:
            sublist=params_inq['sublist']
        except:
             sublist=''
        if descriptor =='num_inq':
            if if_long_description ==0:
                attr_name = "sofi_"+ descriptor + self._get_attr_name_sublist(sublist) + self._get_attr_name_loan_type(loan_type) + self._get_attr_name_window (window, window_unit)
                return attr_name
            else:
                return "number of " + self._get_attr_name_loan_type(loan_type, 1) + self._get_attr_name_sublist(sublist, 1) + " inquiries " + self._get_attr_name_window(window, window_unit, 1)

    def _get_con_loan_type(self,loan_type):
        if loan_type != '':
            return "(case when product_type = '" + loan_type + "' then 1 else 0 end)"
        else:
            return str(1)   

    def _get_con_window(self,window, window_unit):
        if (window != '') & (window_unit != ''):
            if window_unit == 'day':
                day = window
            elif window_unit == 'month':
                day = str(int(window) * 30)
            return "(case when inq_date_dif<=" + day + " then 1 else 0 end)"
        else:
            return str(1)

    def _get_con_sublist(self,sublist):
        if sublist!='':
            return "(case when SubscriberDisplayName in " + sublist + " then 1 else 0 end)"
        else:
            return str(1)

    def _get_con(self,loan_type, window, window_unit, sublist):
        return "sum(" + self._get_con_loan_type(loan_type) + "*" + self._get_con_window(window, window_unit) + "*" + self._get_con_sublist(sublist)

    def get_inq_stats_sql(self):
        params_inq = self.param
        descriptor = params_inq['descriptor']
        source_table = params_inq['source_table']
        try:
            loan_type=params_inq['loan_type']
        except:
             loan_type=''
        try:
            window=params_inq['window']
        except:
             window=''
        try:
            window_unit=params_inq['window_unit']
        except:
             window_unit=''
        try:
            sublist=params_inq['sublist']
        except:
             sublist=''
        if descriptor =='num_inq':
            try:
                field_name = params_inq['field_name']
            except:
                field_name = self.get_attr_name()
            sql = self._get_con(loan_type, window, window_unit, sublist) + ") as " + field_name
        return sql