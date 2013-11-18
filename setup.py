"""Installer for the datagrid package."""

from setuptools import setup


setup(
    name='Datagrid',
    version='0.1',

    # keys: package names (empty name stands for "all packages")
    # values: directory names relative to distribution root where packages
    #         (denoted by <key>) reside
    package_dir = {'datagrid': 'src'},

    # list of packages to process during setup
    packages=['datagrid'],

    # XXX: .git files excluded this way? need to use exclude_package_data?
    include_package_data=True,
    zip_safe=False,

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    # install_requires=[
    #     'setuptools',
    # ],

    # metadata for upload to PyPI
    author='Peter Lamut',
    author_email='',
    description='A small simulation as a part of my Master\'s thesis',
)
