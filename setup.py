from setuptools import setup, find_packages

setup(
    name="common_utils",
    packages=find_packages(include=["common", "common.*"]),
)
