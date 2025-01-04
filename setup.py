import os
import subprocess
from setuptools import setup, find_packages

setup(
    name='phoenix-wrapper',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    description='Python Wrapper for the PHOENIX solver',
    author='Workgroup Stefan Schumacher',
    url='https://github.com/Schumacher-Group-UPB/PHOENIX',
)
