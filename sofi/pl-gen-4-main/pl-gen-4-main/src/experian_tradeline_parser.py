from data import experian_xml_parser


class TradeLineParser(experian_xml_parser.AbstractExperianXmlParser):
    def __init__(self, ignore_closed=False):
        self.ignore_closed = ignore_closed
        super().__init__()

    def parse_xml_string(self, xml_string, id = None,applicant_type = None, credit_pull_id=None, credit_pull_date=None):
        self.parse_xml_dict(experian_xml_parser.parse_experian_xml(xml_string), id, applicant_type, credit_pull_id, credit_pull_date)

    def parse_xml_dict(self, root, id = None, applicant_type = None, credit_pull_id=None, credit_pull_date=None):
        net_connect_response = self._child(root, 'NetConnectResponse')
        products = self._children(net_connect_response, 'Products')
        for product in products:
            custom_solutions = self._children(product, 'CustomSolution')
            for custom_solution in custom_solutions:
                trade_lines = self._children(custom_solution, 'TradeLine')
                for trade_line in trade_lines:
                    if self._should_ignore_trade_line(trade_line):
                        continue
                    self._add_trade_line(trade_line,id, applicant_type, credit_pull_id, credit_pull_date)

    def _should_ignore_trade_line(self, trade_line):
        open_or_closed = self._child_text(trade_line, "OpenOrClosed")
        # Assume missing or empty means open
        return open_or_closed == "Closed" and self.ignore_closed

    def _add_trade_line(self, trade_line,id,applicant_type, credit_pull_id, credit_pull_date):
        row = {}
        row['id'] = id
        row['applicant_type'] = applicant_type
        row['credit_pull_id'] = credit_pull_id
        row['credit_pull_date'] = credit_pull_date
        row['OpenOrClosed'] = self._child_text(trade_line, 'OpenOrClosed')
        row['Subcode'] = self._child_text(trade_line, 'Subcode')
        row['SubscriberDisplayName'] = self._child_text(trade_line, 'SubscriberDisplayName')
        row['AccountNumber'] = self._child_text(trade_line, 'AccountNumber')
        row['AccountTypeCode'] = self._child_attr(trade_line, 'AccountType', 'code')
        row['AccountType'] = self._child_text(trade_line, 'AccountType')
        row['KOBCode'] = self._child_attr(trade_line, 'KOB', 'code')
        row['KOB'] = self._child_text(trade_line, 'KOB')
        row['ECOACode'] = self._child_attr(trade_line, 'ECOA', 'code')
        row['ECOA'] = self._child_text(trade_line, 'ECOA')
        row['RevolvingOrInstallmentCode'] = self._child_attr(trade_line, 'RevolvingOrInstallment', 'code')
        row['RevolvingOrInstallment'] = self._child_text(trade_line, 'RevolvingOrInstallment')
        row['StatusCode'] = self._child_attr(trade_line, 'Status', 'code')
        row['Status'] = self._child_text(trade_line, 'Status')
        row['PaymentProfile'] = self._child_text(trade_line, 'PaymentProfile')
        row['MonthlyPaymentAmount'] = self._child_text(trade_line, 'MonthlyPaymentAmount', cast_type=int)
        row['SubscriberDisplayName'] = self._child_text(trade_line, 'SubscriberDisplayName')
        row['BalanceAmount'] = self._child_text(trade_line, 'BalanceAmount', cast_type=int)
        row['OpenDate'] = self._child_text(trade_line, 'OpenDate', is_experian_date=True)
        row['BalanceDate'] = self._child_text(trade_line, 'BalanceDate', is_experian_date=True)
        row['StatusDate'] = self._child_text(trade_line, 'StatusDate', is_experian_date=True)
        row['TermsDuration'] = self._child_attr(trade_line, 'TermsDuration', 'code')
        row['MonthsHistory'] = self._child_text(trade_line, 'MonthsHistory', cast_type=int)
        row['DelinquenciesOver30Days'] = self._child_text(trade_line, 'DelinquenciesOver30Days', cast_type=int)
        row['DelinquenciesOver60Days'] = self._child_text(trade_line, 'DelinquenciesOver60Days', cast_type=int)
        row['DelinquenciesOver90Days'] = self._child_text(trade_line, 'DelinquenciesOver90Days', cast_type=int)
        row['DerogCounter'] = self._child_text(trade_line, 'DerogCounter', cast_type=int)
        row['AmountPastDue'] = self._child_text(trade_line, 'AmountPastDue', cast_type=int)

        # Enhanced payment data
  
        enhanced_payment_data = self._child(trade_line, 'EnhancedPaymentData')
        row['AccountCondition'] = self._child_text(enhanced_payment_data, 'AccountCondition')
        row['AccountConditionCode'] = self._child_attr(enhanced_payment_data, 'AccountCondition','code')
        row['PaymentStatus'] = self._child_text(enhanced_payment_data, 'PaymentStatus')
        row['PaymentStatusCode'] = self._child_attr(enhanced_payment_data, 'PaymentStatus', 'code')
        row['SpecialComment'] = self._child_text(enhanced_payment_data, 'SpecialComment')
        row['InitialPaymentLevelDate'] = self._child_text(enhanced_payment_data, 'InitialPaymentLevelDate', is_experian_date=True)
        row['EnhancedAccountType'] = self._child_text(enhanced_payment_data, 'AccountType')
        row['EnhancedAccountTypeCode'] = self._child_attr(enhanced_payment_data, 'AccountType', 'code')

        # Amounts
        
        amounts = self._children(trade_line, 'Amount')
        for i in range(0, len(amounts)):
            amount = amounts[i]
            qualifier = self._child_text(amount, "Qualifier")
            value = self._child_text(amount, "Value", cast_type=int)
            row["AmountQualifier_%d" % (i + 1)] = qualifier
            row["AmountValue_%d" % (i + 1)] = value
        #
        # TrendedData:
        # True if the trade line has any <TrendedData>, otherwise False.
        #
        row['HasTrendedData'] = len(self._children(trade_line, 'TrendedData')) != 0
        self._rows.append(row)
