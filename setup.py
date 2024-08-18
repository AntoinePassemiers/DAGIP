# -*- coding: utf-8 -*-
#
#  setup.py
#
#  Copyright 2023 Antoine Passemiers <antoine.passemiers@gmail.com>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.

import os

from setuptools import setup


ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_FOLDER = os.path.join(ROOT, 'dagip')

PROJECT_NAME = 'dagip'

packages = [
    f'{PROJECT_NAME}',
    f'{PROJECT_NAME}.benchmark',
    f'{PROJECT_NAME}.correction',
    f'{PROJECT_NAME}.da',
    f'{PROJECT_NAME}.ichorcna',
    f'{PROJECT_NAME}.nipt',
    f'{PROJECT_NAME}.nn',
    f'{PROJECT_NAME}.optimize',
    f'{PROJECT_NAME}.retraction',
    f'{PROJECT_NAME}.stats',
    f'{PROJECT_NAME}.tools',
    f'{PROJECT_NAME}.transport',
    f'{PROJECT_NAME}.validation'
]

setup(
    name=PROJECT_NAME,
    version='0.0.1',
    description='',
    url='https://github.com/AntoinePassemiers/DAGIP',
    author='Antoine Passemiers',
    packages=packages
)
