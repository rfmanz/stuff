from data import experian_xml_parser


class InquiryParser(experian_xml_parser.AbstractExperianXmlParser):
    def __init__(self):
        super().__init__()

    def parse_xml_dict(self, root, id = None,applicant_type = None, credit_pull_id=None, credit_pull_date=None):
        net_connect_response = self._child(root, 'NetConnectResponse')
        products = self._children(net_connect_response, 'Products')
        for product in products:
            custom_solutions = self._children(product, 'CustomSolution')
            for custom_solution in custom_solutions:
                inquiries = self._children(custom_solution, 'Inquiry')
                for inquiry in inquiries:
                    self._parse_inquiry(inquiry, id, applicant_type, credit_pull_id, credit_pull_date)

    def _parse_inquiry(self, inquiry, id = None,applicant_type = None, credit_pull_id=None, credit_pull_date=None):
        row = {}
        row['id'] =id
        row['applicant_type'] = applicant_type
        row['credit_pull_id'] = credit_pull_id
        row['credit_pull_date'] = credit_pull_date
        # Field parsing goes here
        row['Date'] = self._child_text(inquiry, 'Date', is_experian_date=True)
        row['Amount'] = self._child_text(inquiry, 'Amount', cast_type=int)
        row['TypeCode'] = self._child_attr(inquiry, 'Type', 'code')
        row['Type'] = self._child_text(inquiry, 'Type')
        row['Terms'] = self._child_attr(inquiry, 'Terms', 'code')
        row['Subcode'] = self._child_text(inquiry, 'Subcode')
        row['KOBcode'] = self._child_attr(inquiry, 'KOB', 'code')
        row['KOB'] = self._child_text(inquiry, 'KOB')
        row['SubscriberDisplayName'] = self._child_text(inquiry, 'SubscriberDisplayName')
        # Save the parsed row
        self._rows.append(row)
