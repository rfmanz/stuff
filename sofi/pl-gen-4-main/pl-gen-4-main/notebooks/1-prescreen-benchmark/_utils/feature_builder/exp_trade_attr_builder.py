import re

class exp_trade_attr_builder:
        
    params_0 = {'descriptor': 'num_trade',
                'source_table': 'tradeline',
                'loan_type': '',
                'date_window': '',
                'sublist':"",
                'openorclosed':''}
    
    def __init__(self, param = params_0):
        self.param = param
        
    def _get_attr_name_loan_type(self, loan_type, if_long_description = 0):
        if loan_type != '':
            if if_long_description ==0:
                return '_' + loan_type.split('_')[1]
            else:
                return " " + loan_type.split('_')[1]
        else:
            return ''

    def _get_attr_name_sublist(self, sublist, if_long_description = 0):
        if sublist != '':
            if if_long_description ==0:
                return '_list' 
            else:
                return " " + "lender in list " + sublist
        else:
            return ''

    def _get_attr_name_openorclosed(self, openorclosed, if_long_description = 0):
        if openorclosed != '':
            if if_long_description ==0:
                return '_' + openorclosed.lower()
            else:
                return " " + openorclosed.lower()
        else:
            return ''

    def _get_attr_name_window(self, window, window_unit, if_long_description = 0):
        if (window != '') & (window_unit != ''):
            if if_long_description ==0:
                return '_' + window + window_unit
            else:
                return " " + "in last " + window + " " + window_unit
        else:
            return ''

    def get_attr_name(self, if_long_description = 0):
        params_trade = self.param
        descriptor = params_trade['descriptor']
        source_table = params_trade['source_table']
        try:
            loan_type=params_trade['loan_type']
        except:
            loan_type=''
        try:
            window=re.findall(r'\d+', params_trade['date_window'])[0]
        except:
            window=''
        try:
            window_unit=re.findall("[a-zA-Z]+", params_trade['date_window'])[0]
        except:
            window_unit=''
        try:
            sublist=params_trade['sublist']
        except:
            sublist=''
        try:
            openorclosed=params_trade['openorclosed']
        except:
            openorclosed=''
                
        if if_long_description ==0:
            attr_name = "sofi_" + descriptor + self._get_attr_name_sublist(sublist) + self._get_attr_name_loan_type(loan_type) + self._get_attr_name_openorclosed(openorclosed) + self._get_attr_name_window (window, window_unit)
            return attr_name
        else:
            mapper = {
                "num_trade" : "number of",
                "num_month_recent" : "number of months since most recent",
                "total_bal" : "total balance for",
                "ever_dq" : "ever delinquency on",
                "current_dq" : "if currently delinquency on",
                "avg_term" : "average term for"
            }
            return mapper[descriptor] + self._get_attr_name_sublist(sublist, 1) + " " +  self._get_attr_name_loan_type(loan_type, 1) + " trade " + self._get_attr_name_openorclosed(openorclosed, 1) + self._get_attr_name_window(window, window_unit, 1)           

    def _get_con_loan_type(self, loan_type_flag ):
        if loan_type_flag != '':
            return loan_type_flag
        else:
            return str(1)   

    def _get_con_window(self, window , window_unit):
        if (window != '') & (window_unit != ''):
            return "(case when seasoning_month<=" + window + " then 1 else 0 end)"
        else:
            return str(1)

    def _get_con_openorclosed(self, openorclosed):
        if openorclosed != '':
            return "(case when openorclosed='" + openorclosed + "' then 1 else 0 end)"
        else:
            return str(1)

    def _get_con_sublist(self, sublist):
        if sublist!='':
            return "(case when SubscriberDisplayName in " + sublist + " then 1 else 0 end)"
        else:
            return str(1)

    def _get_con(self, loan_type, window , window_unit, sublist, openorclosed):
        return self._get_con_loan_type(loan_type) + "*" + self._get_con_window(window, window_unit) + "*" + self._get_con_sublist(sublist) + "*" + self._get_con_openorclosed(openorclosed)


    def get_trade_stats_sql(self):
        params_trade = self.param
        descriptor = params_trade['descriptor']
        source_table = params_trade['source_table']
        try:
            loan_type=params_trade['loan_type']
        except:
            loan_type=''
        try:
            window = re.findall(r'\d+', params_trade['date_window'])[0]
        except:
            window=''
        try:
            window_unit=re.findall("[a-zA-Z]+", params_trade['date_window'])[0]
        except:
            window_unit=''
        try:
            sublist=params_trade['sublist']
        except:
            sublist=''
        try:
            openorclosed=params_trade['openorclosed']
        except:
            openorclosed=''          
        try:
            field_name = params_trade['field_name']
        except:
            field_name = self.get_attr_name()   
            
        mapper_operation = {
                "num_trade" : "sum",
                "num_month_recent" : "min",
                "total_bal" : "sum",
                "ever_dq" : "max",
                "current_dq" : "max",
                "avg_term" : "avg"}
        mapper_base = {
                "num_trade" : "1",
                "num_month_recent" : "seasoning_month",
                "total_bal" : "effective_bal",
                "ever_dq" : "flag_everdq",
                "current_dq" : "flag_currentdq",
                "avg_term" : "terms"}
        if descriptor =='num_trade':
            sql_script = "sum("+ self._get_con(loan_type, window, window_unit, sublist, openorclosed) + ") as " + field_name
        else:
            sql_agg = mapper_operation[descriptor] + "(case when " + self._get_con(loan_type, window, window_unit, sublist, openorclosed) + "=1 then " + mapper_base[descriptor] + " else null end)"
            sql_script = "case when sum("+ self._get_con(loan_type, window, window_unit, sublist, openorclosed) + ")>0 then " + sql_agg + " else null end as " + field_name

        return sql_script