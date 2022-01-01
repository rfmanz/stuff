# from setuptools import find_packages, setup
from setuptools import setup

setup(name='pyutils',
      version='0.1',
      description='Basic utilities for data science',
      url='https://github.com/rfmanz/pyutils',
      license='MIT',
#       install_requires=['numpy', 'pandas', 'datatable', 'zipfile','re'],
      install_requires=['numpy', 'pandas', 'datatable','sklearn.preprocessing'],
      packages=['pyutils'])

