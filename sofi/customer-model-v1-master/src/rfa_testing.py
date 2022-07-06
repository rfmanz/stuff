import os
import json

with open('../pyutils/sofi/customer-model-v1-master/config.json', "r") as f:
    CONFIG_FILE = json.load(f)


for name, path in CONFIG_FILE["data"][prefix].items():
    print(name,path)


if "data" not in CONFIG_FILE:
    CONFIG_FILE["data"] = {}

if prefix not in CONFIG_FILE["data"]:
    CONFIG_FILE["data"][prefix] = {}