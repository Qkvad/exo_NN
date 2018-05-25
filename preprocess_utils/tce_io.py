#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:14:17 2018

@author: sandro
"""

import pandas as pd

def read_tce(base_dir, kep_id):
	df = pd.read_csv(base_dir + "dr24_tce.csv", index_col="rowid", comment="#")
	
	return df[df["kepid"] == kep_id]


