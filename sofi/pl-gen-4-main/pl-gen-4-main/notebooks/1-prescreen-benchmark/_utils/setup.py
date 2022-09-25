from setuptools import setup

setup(name = 'utils',
      version = '0.111',
      description = 'Utilities for PL Gen 4 Model',
      packages = ['data','feature_builder','performance_eval', 'feature_selection', 'model_trainer'],
      zip_safe = False)