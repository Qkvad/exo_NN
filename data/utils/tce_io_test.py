#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:29:41 2018

@author: sandro
"""

import tce_io as tceio
from tabulate import tabulate # pip install tabulate

BASE_DIR = "../kepler/"
KEP_ID   = 11442793

tce = tceio.read_tce(BASE_DIR, KEP_ID).T

print(tabulate(tce, headers='keys', tablefmt='psql'))
