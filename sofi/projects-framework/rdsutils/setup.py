import os
from setuptools import setup, find_packages

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")) as f:
    readme = f.read()

version = "0.0.14"
package_name = "rdsutils"

requirements = [
    "lightgbm>=2.2.3" "boto3>=1.9.0",
    "pandas>0.22.0",
    "matplotlib",
    "psycopg2-binary",
    "seaborn",
    "SQLAlchemy>=1.2.11",
    "colorama",
]

setup(
    name=package_name,
    version=version,
    author="SoFi",
    author_email="tboser@sofi.org",
    description="Risk data science convenience package.",
    packages=find_packages(),
    license="proprietary",
    long_description=readme,
    install_requires=requirements,
    keywords="machine-learning",
    entry_points={"console_scripts": []},
    extras_require={},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: Proprietary :: All Rights Reserved by SoFi Inc.",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.9",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Operating System :: POSIX",
    ],
)
