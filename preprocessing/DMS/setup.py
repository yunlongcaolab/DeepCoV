import os

from setuptools import setup, find_packages

pkg_name = "dms"

_cur_dir = os.path.dirname(__file__)

setup(
    name=pkg_name,
    version="0.1.0",
    include_package_data=True,
    packages=find_packages()
)
