"""
Reporting util functions
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import datetime
import json
import os
import re

import numpy as np
import pandas as pd


# general json functions
# ======================

class GRMlabEncoder(json.JSONEncoder):
	"""GRMlab style JSON encoder."""
	def default(self, obj):

		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.float):
			if np.isnan(obj):
				return None
			else:
				return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		elif isinstance(obj, list):
			return [None if pd.isnull(x) else x for x in obj]
		else:
			return super(GRMlabEncoder, self).default(obj)


def dict_to_json(dictionary):
	"""Convert python dictionary to json string."""
	return json.dumps(dictionary, separators=(',', ':'), cls=GRMlabEncoder)


# reporting functions
# ===================

def item_steps_element(step_id, step_type, step_contents):
	json_item_steps = {
		"step_id": step_id,
		"step_type": step_type,
		"step_contents": step_contents
	}

	return json_item_steps


def content_blocks_element(block_type, block_data, is_json=False):
	"""
	Generate content block element as a dictionary. If block_data is a
	pandas json output => deserialize.
	"""
	bk_data = block_data
	if is_json:
		bk_data = json.loads(block_data)

	json_content_blocks = {
		"block_type": block_type,
		"block_data": bk_data
	}

	return json_content_blocks


def step_contents_element(content_position, content_type, content_blocks):
	"""Generate step content element as a dictionary."""
	json_step_contents = {
		"content_position": content_position,
		"content_type": content_type,
		"content_blocks": content_blocks
	}

	return json_step_contents


def results_tables(tables, table_names):
	"""Handle cases with multiple results tables."""
	if table_names is not None:
		if len(tables) != len(table_names):
			raise ValueError("tables and table_names different size.")
	else:
		table_names = ["table_{}".format(i) for i in range(len(tables))]

	return [content_blocks_element(block_type=table_names[i],
		block_data=tables[i], is_json=True) for i in range(len(tables))]


def get_id(file):
	"""
	Read last id and increment it. CUrrent implementation is not optimal for 
	large files.
	"""
	with open(file, 'r') as f:		
		last_line = f.read().splitlines()[-1]

	valid_last_line = ("json_items.push" in last_line)

	if valid_last_line:
		regex = re.compile(r'"item_id":"\d{4}"')
		id = int("".join(s for s in regex.findall(last_line)[0] if s.isdigit()))
		str_new_id = str(id + 1)
		return "0"*(4 - len(str_new_id))+str_new_id
	else:
		# first item in contents.json
		return "0001"


def is_report(path, system_os, verbose=True):
	"""Check whether directory is a grmlab report."""
	if system_os is "Windows":
		contents_path = path + "metadata\\contents.json"
	else:
		contents_path = path + "metadata/contents.json"

	try:
		# catch if file does not exist => new report.
		with open(contents_path, 'r') as f:
			first_line = [next(f) for _ in range(2)][1]

		valid_first_line = ("user_info" in first_line)

		if valid_first_line:
			regex1 = '(?:"(user_id|user_domain|user_machine|datetime)":")(.*?)(?:")'
			regex = re.compile(regex1)
			user_info = regex.findall(first_line)

			# print current report information
			if verbose:
				report_user_info = (
					"Path: {} \n\n"
					"Selected folder is a GRMlab report created by user\n"
					"+-------------+----------------------------------------+\n"
					"|{:<13}| {:>38} |\n"
					"|{:<13}| {:>38} |\n"
					"|{:<13}| {:>38} |\n"
					"|{:<13}| {:>38} |\n"
					"+-------------+----------------------------------------+\n"
					).format(path, *sum(user_info, ()))
				print(report_user_info)
			return True
	except:
		return False


def user_info():
	"""Return user information and current datetime."""
	env = os.environ
	try:
		info = {"value_hosts": env["VALUE_HOSTS"],
			"uuaa_code": env["UUAA_CODE"],
			"datetime": str(datetime.datetime.now())}
	except:
		info = {"value_hosts": None,
			"uuaa_code": None,
			"datetime": str(datetime.datetime.now())}
	return info


def reporting_class_type_check(argname, arg, grmlab_class):
	"""
	Return error message for reporting class argument when not being of a class
	in grmlab.
	"""
	if not isinstance(arg, grmlab_class):
		msg_err_argtype = "Argument {} must be a class of type {}."
		raise TypeError(msg_err_argtype.format(argname, grmlab_class.__name__))


def reporting_output_format(df, format, index=False):
	"""Return pandas.DataFrame or JSON file."""
	if not format in ("dataframe", "json"):
		raise ValueError("format {} not supported.".format(format))

	if not isinstance(df, pd.DataFrame):
		raise TypeError("df must be a pandas.DataFrame.")

	if format is "dataframe":
		return df

	elif format is "json":
		return df.to_json(orient="split", index=index)

	return df
