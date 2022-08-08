"""
GRMlab reporting functions.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import os
import platform
import shutil
import sys

from ..core.base import GRMlabBase
from .util import dict_to_json
from .util import get_id
from .util import is_report
from .util import user_info
from .util import content_blocks_element
from .util import item_steps_element
from .util import results_tables
from .util import step_contents_element


# JSON code
JSON_CONTENTS_USER_INFO = "var user_info = '{}';"
JSON_CONTENTS_REPORT_INFO = "var report_info = '{}';"
JSON_CONTENTS_ITEMS_ARRAY = "var json_items = new Array();"
JSON_CONTENTS_ITEMS = "json_items.push('{}');"
JSON_DATA_ITEMS_OBJECT = "var json_items_data = new Object();"
JSON_DATA_ITEMS = "json_items_data['{}'] = ('{}');"


class Reporting(GRMlabBase):
    """
    GRMlab reporting.

    Generate GRMlab reporting folder.

    Parameters
    ----------
    path : str
        The path to create a new reporting folder.

    title : str or None (default=None)
        The title of the report.

    description : str or None (default=None)
        The description of the report.

    date : str or None (default=None)
        The date of the report.

    verbose : int or boolean (default=False)
        Controls verbosity of output.
    """
    def __init__(self, path, title=None, description=None, date=None,
                 verbose=False):

        self.path = path if path is not None else ""
        self.title = title if title is not None else ""
        self.description = description if description is not None else ""
        self.date = date
        self.verbose = verbose

        # main json files
        self._json_contents_file = None
        self._json_data_file = None

        # initialize report
        self._init_report()

    def add(self, grmlabcls):
        """
        Add GRMlab class to reporting.

        Parameters
        ----------
        grmlabcls : object
            A supported GRMlab class.
        """
        class_name = grmlabcls.__class__.__name__

        if class_name == "PreprocessingDataStream":
            from .preprocessing import add_preprocessing_data_stream

            item_info = add_preprocessing_data_stream(grmlabcls)
            self._build_item(**item_info)
        elif class_name in ["Preprocessing", "PreprocessingSpark"]:
            from .preprocessing import add_preprocessing

            item_info = add_preprocessing(grmlabcls)
            self._build_item(**item_info)
        elif class_name in ["Univariate", "UnivariateSpark"]:
            from .univariate import add_univariate
            from .univariate import add_univariate_variable

            item_info = add_univariate(grmlabcls)
            item_id = self._build_item(**item_info)

            # build item for each variable in univariate
            variables_info = [
                add_univariate_variable(variable, info, item_id,
                                        grmlabcls._unique_dates)
                for variable, info in grmlabcls._dict_variables.items()]

            # add variables to json
            self._add_lines("data", variables_info)
        elif class_name == "Bivariate":
            from .bivariate import add_bivariate
            from .bivariate import add_bivariate_variable

            item_info = add_bivariate(grmlabcls)
            item_id = self._build_item(**item_info)

            # build item for each variable in bivariate
            variables_info = [
                add_bivariate_variable(variable, info, item_id,
                                       grmlabcls._unique_dates)
                for variable, info in grmlabcls._dict_variables.items()]

            # add variables to json
            self._add_lines("data", variables_info)
        elif class_name == "MultivariateAnalysis":
            from .multivariate_analysis import add_multivariate
            from .multivariate_analysis import add_multivariate_variable
            from .multivariate_analysis import add_multivariate_group

            item_info = add_multivariate(grmlabcls)
            item_id = self._build_item(**item_info)

            # build item for each variable in multivariate
            variables_info = [
                add_multivariate_variable(variable, info, item_id, grmlabcls)
                for variable, info in grmlabcls._dict_variables.items()]

            # build item for each group in multivariate
            groups_info = [
                add_multivariate_group(group, info, item_id, grmlabcls)
                for group, info in grmlabcls._dict_groups.items()]

            # add variables to json
            self._add_lines("data", variables_info + groups_info)

        elif class_name == "BivariateContinuous":
            from .bivariate_continuous import add_bivariate
            from .bivariate_continuous import add_bivariate_variable

            item_info = add_bivariate(grmlabcls)
            item_id = self._build_item(**item_info)

            # build item for each variable in bivariate
            variables_info = [
                add_bivariate_variable(variable, info, item_id,
                                       grmlabcls._unique_dates)
                for variable, info in grmlabcls._dict_variables.items()]

            # add variables to json
            self._add_lines("data", variables_info)
        elif class_name == "ModelGenerator":
            from .modelgenerator import add_model_generator

            item_info = add_model_generator(grmlabcls)
            self._build_item(**item_info)
        elif class_name == "ModelOptimizer":
            from .model_optimization import add_model_optimizer

            # whole content of analysis within one content_block
            step_contents = add_model_optimizer(grmlabcls, "")

            item_steps = [item_steps_element(step_id="01", step_type="run",
                                             step_contents=[step_contents])]
            item_type = grmlabcls.__class__.__name__.lower()

            # item structure
            item_id, json_item = self._item_element(
                item_type=item_type, item_layout_version="02",
                item_steps=item_steps
                )
            # structure of json for file contents
            json_content = {
                "item_id": json_item["item_id"],
                "item_info": 0,
                "step_id": "00",
                "title": "Model Optimizer",
                "type": json_item["item_type"]
            }
            str_json_content = JSON_CONTENTS_ITEMS.format(
                dict_to_json(json_content))

            # add line to contents file
            self._add_line("contents", str_json_content)
            # add line to data file
            self._add_item_data(json_item)
        elif class_name == "MultivariateCorrelations":
            from .multivariate_analysis import add_corr_analysis

            # whole content of analysis within one content_block
            step_contents = add_corr_analysis(grmlabcls, "")

            item_steps = [item_steps_element(step_id="01", step_type="run",
                                             step_contents=[step_contents])]
            item_type = grmlabcls.__class__.__name__.lower()

            # item structure
            item_id, json_item = self._item_element(
                item_type=item_type, item_layout_version="02",
                item_steps=item_steps
                )
            # structure of json for file contents
            json_content = {
                "item_id": json_item["item_id"],
                "item_info": 0,
                "step_id": "00",
                "title": "Multivariate Correlation",
                "type": json_item["item_type"]
            }
            str_json_content = JSON_CONTENTS_ITEMS.format(
                dict_to_json(json_content))

            # add line to contents file
            self._add_line("contents", str_json_content)
            # add line to data file
            self._add_item_data(json_item)
        elif class_name == "ModelAnalyzer":
            from .model_analysis import add_model_analysis

            # whole content of analysis within one content_block
            step_contents = add_model_analysis(grmlabcls, "")

            item_steps = [item_steps_element(step_id="01", step_type="run",
                                             step_contents=[step_contents])]
            item_type = grmlabcls.__class__.__name__.lower()

            # item structure
            item_id, json_item = self._item_element(
                item_type=item_type, item_layout_version="02",
                item_steps=item_steps
                )
            # structure of json for file contents
            json_content = {
                "item_id": json_item["item_id"],
                "item_info": 0,
                "step_id": "00",
                "title": "Model Analysis",
                "type": json_item["item_type"]
            }
            str_json_content = JSON_CONTENTS_ITEMS.format(
                dict_to_json(json_content))

            # add line to contents file
            self._add_line("contents", str_json_content)
            # add line to data file
            self._add_item_data(json_item)
        elif class_name == "ModelAnalyzerContinuous":
            from .model_analysis import add_model_analysis_continuous

            # whole content of analysis within one content_block
            step_contents = add_model_analysis_continuous(grmlabcls, "")

            item_steps = [item_steps_element(step_id="01", 
                step_type="run", step_contents=[step_contents])]
            item_type = grmlabcls.__class__.__name__.lower()

            # item structure
            item_id, json_item = self._item_element(
                item_type=item_type, item_layout_version="02", 
                item_steps=item_steps
                )
            # structure of json for file contents
            json_content = {
                "item_id": json_item["item_id"],
                "item_info": 0,
                "step_id": "00",
                "title": "Model Analysis",
                "type": json_item["item_type"]
            }
            str_json_content = JSON_CONTENTS_ITEMS.format(
                dict_to_json(json_content))

            # add line to contents file
            self._add_line("contents", str_json_content)
            # add line to data file
            self._add_item_data(json_item)
        elif class_name == "ModelComparison":
            from .model_comparison import add_model_comparison

            # whole content of analysis within one content_block
            step_contents = add_model_comparison(grmlabcls, "")

            item_steps = [item_steps_element(step_id="01", step_type="run",
                                             step_contents=[step_contents])]
            item_type = grmlabcls.__class__.__name__.lower()

            # item structure
            item_id, json_item = self._item_element(
                item_type=item_type, item_layout_version="02",
                item_steps=item_steps
                )
            # structure of json for file contents
            json_content = {
                "item_id": json_item["item_id"],
                "item_info": 0,
                "step_id": "00",
                "title": "Model Comparison",
                "type": json_item["item_type"]
            }
            str_json_content = JSON_CONTENTS_ITEMS.format(
                dict_to_json(json_content))

            # add line to contents file
            self._add_line("contents", str_json_content)
            # add line to data file
            self._add_item_data(json_item)
        else:
            raise ValueError("GRMlab class not supported.")

        return self

    def _add_item_contents(self, json_item, item_title):
        """Add item to file contents.json."""
        item_id = json_item["item_id"]

        # main item
        json_content = {
            "item_id": item_id,
            "item_info": 0,
            "step_id": "00",
            "title": item_title,
            "type": json_item["item_type"]
        }

        # subitems (run/transform etc...)
        str_json_content = JSON_CONTENTS_ITEMS.format(
            dict_to_json(json_content))
        self._add_line("contents", str_json_content)

        # each item step adds a new line in contents.json
        for item_step in json_item["item_steps"]:
            json_content = {
                "item_id": item_id,
                "step_id": item_step["step_id"],
                "title": item_step["step_type"].capitalize(),
                "type": item_step["step_type"]
            }

            str_json_content = JSON_CONTENTS_ITEMS.format(
                dict_to_json(json_content))
            self._add_line("contents", str_json_content)

    def _add_item_data(self, json_item):
        """Add item to file data.json."""
        id = json_item["item_id"]
        str_json_data = JSON_DATA_ITEMS.format(id, dict_to_json(json_item))
        self._add_line("data", str_json_data)

    def _add_line(self, file, line):
        """Add line to json file."""
        if file == "data":
            file = self._json_data_file
        elif file == "contents":
            file = self._json_contents_file

        try:
            with open(file, 'a') as f:
                f.write(line+"\n")
        except EnvironmentError as err:
            print("Error: {}".format(err))
            raise
        except:
            print("Unexpected error: {}".format(sys.exc_info()[0]))
            raise

    def _add_lines(self, file, lines):
        """Add several lines at once to json file."""
        if file == "data":
            file = self._json_data_file
        elif file == "contents":
            file = self._json_contents_file

        try:
            with open(file, 'a') as f:
                for line in lines:
                    f.write(line+"\n")
        except EnvironmentError as err:
            print("Error: {}".format(err))
            raise
        except:
            print("Unexpected error: {}".format(sys.exc_info()[0]))
            raise

    def _build_item(self, grmlabcls, item_title, step_config_run=None,
                    step_config_transform=None, step_stats_run=None,
                    step_stats_transform=None, results_run_names=None,
                    results_transform_names=None, extra_item_step=None):

        """Build item information."""
        if grmlabcls._is_run:
            item_steps = []
            step_run_contents = []

            # retrieve information from GRMlab class input parameters
            if step_config_run is not None:
                step_run_contents.append(step_config_run)

            # retrieve information from GRMlab class method run()
            if step_stats_run is not None:
                step_run_contents.append(step_stats_run)

            # retrieve information from GRMlab class method results(step="run")
            # convert pandas json to content block structure
            results_data = grmlabcls.results(step="run", format="json")

            if isinstance(results_data, (tuple, list)):
                _results_tables = results_tables(results_data,
                                                 results_run_names)
            else:
                _results_tables = [content_blocks_element(
                    block_type="table", block_data=results_data, is_json=True)]

            contents_results = step_contents_element(
                content_position="mainbody", content_type="summary",
                content_blocks=_results_tables)
            step_run_contents.append(contents_results)

            # build item step
            item_run = item_steps_element(step_id="01", step_type="run",
                                          step_contents=step_run_contents)
            item_steps.append(item_run)

        if hasattr(grmlabcls, '_is_transformed') and grmlabcls._is_transformed:
            step_transform_contents = []

            # retrieve information from GRMlab class input parameters
            if step_config_transform is not None:
                step_transform_contents.append(step_config_transform)

            # retrieve information form class method transform
            if step_stats_transform is not None:
                step_transform_contents.append(step_stats_transform)

            # retrieve information from GRMlab class method
            # results(step="transform") convert pandas json to content block
            # structure
            results_data = grmlabcls.results(step="transform", format="json")

            if isinstance(results_data, (tuple, list)):
                _results_tables = results_tables(results_data,
                                                 results_transform_names)
            else:
                _results_tables = [content_blocks_element(
                    block_type="table", block_data=results_data, is_json=True)]

            contents_results = step_contents_element(
                content_position="mainbody", content_type="summary",
                content_blocks=_results_tables)
            step_transform_contents.append(contents_results)

            # build item step
            item_transform = item_steps_element(
                step_id="02", step_type="transform",
                step_contents=step_transform_contents)
            item_steps.append(item_transform)

        # build item element
        item_type = grmlabcls.__class__.__name__.lower()
        item_id, json_item = self._item_element(
            item_type=item_type, item_layout_version="02",
            item_steps=item_steps)

        # add items to data.json and contents.json
        self._add_item_data(json_item)
        self._add_item_contents(json_item, item_title)

        # by default return item_id => needed for individual json variable
        return item_id

    def _init_json_data(self):
        """Initialize JSON files: data.json and contents.json."""
        self._add_line("data", JSON_DATA_ITEMS_OBJECT)
        self._add_line("contents", JSON_CONTENTS_REPORT_INFO.format(
            dict_to_json(self._report_info())))
        self._add_line("contents", JSON_CONTENTS_USER_INFO.format(
            dict_to_json(user_info())))
        self._add_line("contents", JSON_CONTENTS_ITEMS_ARRAY)

    def _init_report(self):
        """Initialize GRMlab report."""

        # path format according to o.s.
        abspath = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
        system_os = platform.system()
        linux_os = (system_os == "Linux" or "CYGWIN" in system_os)

        if system_os == "Windows":
            # NOTE: ananconda accepts linux style paths, but we cannot access
            # files without proper windows paths.
            dest = self.path.replace("/", "\\")
        else:
            dest = self.path

        # check if folder structure already corresponds to grmlab report
        path_is_report = is_report(dest, system_os)

        if not path_is_report:
            # new report
            if linux_os:
                source = abspath+"/reporting/report/"
            else:
                source = abspath+"\\reporting\\report\\"

            # generate list of files to be copied
            shutil.copytree(source, dest, copy_function=shutil.copy)

        # save paths to contents.json and data.json
        if linux_os:
            self._json_data_file = dest+"/metadata/data.json"
            self._json_contents_file = dest+"/metadata/contents.json"
        else:
            self._json_data_file = dest+"\\metadata\\data.json"
            self._json_contents_file = dest+"\\metadata\\contents.json"

        # initialize json files
        if not path_is_report:
            self._init_json_data()

    def _item_element(self, item_type, item_layout_version, item_steps):
        """Return item element json and corresponding item_id."""
        item_id = get_id(self._json_contents_file)
        json_item = {
            "item_id": item_id,
            "item_type": item_type,
            "item_layout_version": item_layout_version,
            "item_steps": item_steps
        }

        return item_id, json_item

    def _report_info(self):
        """Return report information summary."""
        report_info = {
            "title1": self.title,
            "title2": self.description,
            "title3": self.date
        }

        return report_info
