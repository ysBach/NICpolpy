"""
"""

from setuptools import setup, find_packages

setup_requires = []
install_requires = ['numpy',
                    'pandas',
                    'photutils >= 0.7',
                    'astropy >= 5.0',
                    'ccdproc >= 1.3']

classifiers = ["Intended Audience :: Science/Research",
               "Operating System :: OS Independent"]

setup(
    name="nicpolpy",
    version="0.1.dev",
    author="Yoonsoo P. Bach",
    author_email="ysbach93@gmail.com",
    description="",
    license="",
    keywords="",
    url="",
    classifiers=classifiers,
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=install_requires)
