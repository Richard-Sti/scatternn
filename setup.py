# Copyright (C) 2022 Richard Stiskalek
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from setuptools import (setup, find_packages)

with open('README.md', "r") as fh:
    long_description = fh.read()

# get version
with open("gausslossnn/_version.py", "r") as fh:
    vstr = fh.read().strip()
try:
    vstr = vstr.split('=')[1].strip()
except IndexError:
    raise RuntimeError("version string in empiricalgalo._verion.py not "
                       "formatted correctly; it should be:\n"
                       "__version__ = VERSION")

with open("requirements.txt", "r") as fh:
    requirements = fh.read()
requirements.split("\n")
# remove ' and " from version string
version = vstr.replace("'", "").replace('"', "")

setup(
    name='GaussLossNN',
    version=version,
    description='NN with a Gaussian loss function.',
    long_description=long_description,
    url='https://github.com/Richard-Sti/empiricalgalo',
    author='Richard Stiskalek',
    author_email='richard.stiskalek@protonmail.com',
    license='GPL-3.0',
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        'Intended Audience :: Science/Research',
        'Natural Language :: English'],
)
