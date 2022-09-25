import pandas as pd
import xmltodict


def parse_experian_xml(xml_string):
    """Convert an XML string to a dictionary using xmltodict."""
    namespaces = {
        'http://www.experian.com/NetConnectResponse': None,
        'http://www.experian.com/ARFResponse': None,
        'http://www.experian.com/CPUResponse': None,
    }
    try:
        root = xmltodict.parse(xml_string, process_namespaces=True, namespaces=namespaces)
        return root
    except :
        print("XML Parse Error\n")
        return -99


class AbstractExperianXmlParser():
    """Base for classes that work with Experian XML credit pulls."""
    def __init__(self):
        self._rows = []

    def to_data_frame(self):
        """Convert self._rows to a Pandas data frame."""
        all_keys = {}
        for row in self._rows:
            for key in row.keys():
                all_keys[key] = True
        df = pd.DataFrame(data=self._rows, columns=all_keys.keys())
        return df

    def _children(self, element, child_name):
        """Return an array of all children with the given tag name."""
        if element is None:
            return None
        children = element[child_name] if child_name in element else []
        return children if isinstance(children, list) else [children]

    def _child(self, element, child_name):
        """Given an element return its only child with the given tag name, or None if it doesn't have one."""
        child = element[child_name] if child_name in element else None
        if isinstance(child, list):
            raise Exception("Expected one child named %s but found %s" % (child_name, len(child)))
        return child

    def _child_attr(self, element, child_name, attr_name, cast_type=str, is_experian_date=False):
        """Given an element get one of its children and return its attribute."""
        child = self._child(element, child_name)
        return self._attr(child, attr_name, cast_type, is_experian_date)

    def _child_text(self, element, child_name, cast_type=str, is_experian_date=False):
        """Given an element get one of its children and return its text."""
        child = self._child(element, child_name)
        return self._text(child, cast_type, is_experian_date)

    def _attr(self, element, attr_name, cast_type=str, is_experian_date=False):
        """Given an element return a given attribute"""
        if element is None:
            return None
        attr_name = '@' + attr_name
        if attr_name not in element:
            return None
        return self._format_string(element[attr_name], cast_type, is_experian_date)

    def _text(self, element, cast_type=str, is_experian_date=False):
        """Given an element return its text"""
        if element is None:
            return None
        string_value = None
        if isinstance(element, str):
            string_value = element
        elif '#text' in element:
            string_value = element['#text']
        return self._format_string(string_value, cast_type, is_experian_date)

    def _format_string(self, string_value, cast_type=str, is_experian_date=False):
        """Clean up a string value from an Experian XML attribute or text."""
        if string_value is None:
            return None
        string_value = string_value.strip()
        if len(string_value) == 0:
            return None
        if is_experian_date:
            string_value = self._format_experian_date(string_value)
        try:
            return cast_type(string_value)
        except ValueError:
            return None

    def _format_experian_date(self, string):
        """Convert a MMDDYYYY date string into YYYY-MM-DD. """
        if string == None or len(string) != 8:
            return string
        else:
            return string[4:8] + "-" + string[0:2] + "-" + string[2:4]
