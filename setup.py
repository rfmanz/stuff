from setuptools import find_packages, setup

setup(name='pyutils',
      version='0.1',
      description='Basic utilities for data science',
      url='https://github.com/rfmanz/pyutils',
      license='MIT',
      packages=find_packages(where="pyutils"),
      package_dir={"": "pyutils"},
      zip_safe=False)


