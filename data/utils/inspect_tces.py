#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 18:23:42 2018

@author: sandro
"""

## STRANGE: 7659389, 12019440, 1026032, 2860594, 4348313, 2708156, 11287726, 9024857, 7597703, 9541567, ! 8308347 !, 8009500, 4064365, 7628336, ! 10464666 !, ! 7128918 !, ? 9659036 ?, 3869825, 3955867
## NTP
# 6426158
## UNK
# 8572936, 8758161
#
# COOL: 9958053

import pandas as pd
import numpy as np
import sklearn

from tabulate import tabulate
import sys

import kepler_io_test as kepio
import tce_io as tceio

# ARGUMENTS:
#  (str)  argv[1]: label in {PC,NTP,AFP,UNK}
#  (bool) argv[2]: shuffle or not
#  (int)  argv[3]: index to continue from
#  (int)  argv[4]: kepid to show just one, -1 otherwise
#  (int)  argv[5]: -p to show processed lightcurve, anything otherwise

df = pd.read_csv("data/csv/dr24_tce.csv", index_col="rowid", comment="#")
print('df shape: ', df.shape)

proc = str(sys.argv[5])

just_one = int(sys.argv[4])
if just_one != -1: 
	df = df[df["kepid"] == int(sys.argv[4])]
	df.tce_duration/=24
	kep_id = df.iloc[0]['kepid']
	print(tabulate(df[['kepid','av_training_set','tce_period','tce_time0bk','tce_duration','tce_plnt_num','tce_ror','tce_dor','tce_depth','tce_eqt','tce_nkoi']].T, headers='keys', tablefmt='psql'))	
	if proc=='-p':
		kepio.show_processed_lightcurve(kep_id)
	else:
		kepio.show_lightcurve(kep_id)

else:
	df_UNK = df[df["av_training_set"] == str(sys.argv[1])]

	shuffle=bool(sys.argv[2])
	if shuffle:
		df_UNK = sklearn.utils.shuffle(df_UNK)

	continue_from = int(sys.argv[3])
	for i in range(0,df_UNK.shape[0]):
		if i<continue_from and not shuffle:
			continue
		print('= '+str(i)+' =')
		kep_id = df_UNK.iloc[i]['kepid']
		tce = tceio.read_tce("data/csv/", kep_id)
		tce.tce_duration/=24
		print(tabulate(tce[['kepid','av_training_set','tce_period','tce_time0bk','tce_duration','tce_plnt_num','tce_ror','tce_dor','tce_depth','tce_eqt','tce_nkoi']].T, headers='keys', tablefmt='psql'))
		if proc=='-p':
			kepio.show_processed_lightcurve(kep_id)
		else:
			kepio.show_lightcurve(kep_id)


