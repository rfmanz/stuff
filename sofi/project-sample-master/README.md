# sample-project-repo

---

This is a template repo for the Risk Data Science team. Please feel free to reach out and contribute to this setup!

Example Project:
* [money-ach-model](https://gitlab.com/sofiinc/data-science-risk/sofi-money-risk-models/deposit-risk-model-v2-ach)

### Tools used
* Git
* Conda
* Markdown
* Bash
* Python

### Core:
* README.md 
    * use: project introduction, usually contains a quick summary of the project purpose and instruction to run script.
    * example: [money-customer-model](https://gitlab.com/sofiinc/data-science-risk/money-risk-models/-/tree/master/money-customer-risk)
    * It uses Markdown language, same as jupyter notebooks.
* requirements.txt
    * use: to install all required packages for this project.
    * how to install: `pip install -i https://repository.sofi.com/artifactory/api/pypi/pypi/simple -r requirements.txt`
    * note: in most cases, we can install most packages by `pip install -r requirements.txt`. However often times we want to use SoFi customized library/package, which is why we need to insert the `-i https://repository.sofi.com/artifactory/api/pypi/pypi/simple` portion.
    * a couple useful internal packages: rdsutils, mdsutils
* .gitignore
    * ignore the file and folders we do not want to store in git due to size.

### Optional & Advanced but highly recommended:
* Makefile
    * a useful tool to do some simple automation. 
    * manual: make [argument]
    * example: 
        * `make requirements` will install packages (from requirements) in this current environment.
        * `make environment` will build a fresh environment, parameterized by the `env.sh` file.
* config.json
    * a json file that contains all hyperparameters during model development.
    * purpose: all notebooks and python scripts read in this json and build model based on the hyperparameters included. Easy for housekeeping.
* folder organization (purly lifestyle)
    * notebooks: directory to for all jupyter notebooks.
    * src: all script related to this project.
    * artifacts: work in progress of model development, such as processed data, visualizations, and statistics, etc.
    * data: data needed. 
        * note: it's very convenient to store everything on S3 and load directly using pandas.
* main.py
    * use: gateway to running the project. 
    * example: [money-customer-model](https://gitlab.com/sofiinc/data-science-risk/money-risk-models/-/tree/master/money-customer-risk)
