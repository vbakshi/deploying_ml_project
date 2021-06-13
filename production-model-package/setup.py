
from pathlib import Path 
from setuptools import find_packages, setup 

# Package meta-data 

NAME = 'regression-model'
DESCRIPTION = " Regression model for predicting house prices"
AUTHOR = "VinayakBakshi"
EMAIL = "vinayakbakshi91@gmail.com"
REQUIRES_PYTHON = ">=3.6.0"

about = {}
ROOT_DIR = Path(__file__).resolve().parent 
REQUIREMENTS_DIR = ROOT_DIR / "requirements"
PACKAGE_DIR = ROOT_DIR / "regression_model"

with open(PACKAGE_DIR / "VERSION") as f:

    _version = f.read().strip
    about['__version__'] = _version 

print(PACKAGE_DIR)

setup(
    name=NAME,
    version='0.0.1',
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=("tests",)),
    package_data={"regression_model": ["VERSION"]},
    # install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license="BSD-3",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)