from data import experian_xml_parser


class ProfileSummaryParser(experian_xml_parser.AbstractExperianXmlParser):
    def __init__(self):
        super().__init__()

    def parse_xml_dict(self, root, id = None,applicant_type = None, credit_pull_id=None, credit_pull_date=None):
        net_connect_response = self._child(root, 'NetConnectResponse')
        products = self._children(net_connect_response, 'Products')
        for product in products:
            custom_solutions = self._children(product, 'CustomSolution')
            for custom_solution in custom_solutions:
                profile_summaries = self._children(custom_solution, 'ProfileSummary')
                for profile_summary in profile_summaries:
                    self._parse_profile_summary(profile_summary, id, applicant_type, credit_pull_id, credit_pull_date)

    def _parse_profile_summary(self, profile_summary, id = None,applicant_type = None, credit_pull_id=None, credit_pull_date=None):
        row = {}
        row['id'] = id
        row['applicant_type'] = applicant_type
        row['credit_pull_id'] = credit_pull_id
        row['credit_pull_date'] = credit_pull_date
        # Field parsing goes here
        row['DisputedAccountsExcluded'] = self._child_text(profile_summary, 'DisputedAccountsExcluded', cast_type=int)
        row['PublicRecordsCount'] = self._child_text(profile_summary, 'PublicRecordsCount', cast_type=int)
        row['InstallmentBalance'] = self._child_text(profile_summary, 'InstallmentBalance', cast_type=int)
        row['RealEstateBalance'] = self._child_text(profile_summary, 'RealEstateBalance', cast_type=int)
        row['RevolvingBalance'] = self._child_text(profile_summary, 'RevolvingBalance', cast_type=int)
        row['PastDueAmount'] = self._child_text(profile_summary, 'PastDueAmount', cast_type=int)
        row['MonthlyPayment'] = self._child_text(profile_summary, 'MonthlyPayment', cast_type=int)
        row['MonthlyPaymentPartialFlag'] = self._child_text(profile_summary, 'MonthlyPaymentPartialFlag', cast_type=int)
        row['RealEstatePayment'] = self._child_text(profile_summary, 'RealEstatePayment', cast_type=int)
        row['RealEstatePaymentPartialFlag'] = self._child_text(profile_summary, 'RealEstatePaymentPartialFlag', cast_type=int)
        row['RevolvingAvailablePercent'] = self._child_text(profile_summary, 'RevolvingAvailablePercent', cast_type=int)
        row['RevolvingAvailablePartialFlag'] = self._child_text(profile_summary, 'RevolvingAvailablePartialFlag', cast_type=int)
        row['TotalInquiries'] = self._child_text(profile_summary, 'TotalInquiries', cast_type=int)
        row['InquiriesDuringLast6Months'] = self._child_text(profile_summary, 'InquiriesDuringLast6Months', cast_type=int)
        row['TotalTradeItems'] = self._child_text(profile_summary, 'TotalTradeItems', cast_type=int)
        row['PaidAccounts'] = self._child_text(profile_summary, 'PaidAccounts', cast_type=int)
        row['SatisfactoryAccounts'] = self._child_text(profile_summary, 'SatisfactoryAccounts', cast_type=int)
        row['NowDelinquentDerog'] = self._child_text(profile_summary, 'NowDelinquentDerog', cast_type=int)
        row['WasDelinquentDerog'] = self._child_text(profile_summary, 'WasDelinquentDerog', cast_type=int)
        row['OldestTradeOpenDate'] = self._child_text(profile_summary, 'OldestTradeOpenDate', is_experian_date=True)
        row['DelinquenciesOver30Days'] = self._child_text(profile_summary, 'DelinquenciesOver30Days', cast_type=int)
        row['DelinquenciesOver60Days'] = self._child_text(profile_summary, 'DelinquenciesOver60Days', cast_type=int)
        row['DelinquenciesOver90Days'] = self._child_text(profile_summary, 'DelinquenciesOver90Days', cast_type=int)
        row['DerogCounter'] = self._child_text(profile_summary, 'DerogCounter', cast_type=int)

        # Save the parsed row
        self._rows.append(row)
