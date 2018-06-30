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
# 8572936, 8758161, 3239945
#
# COOL: 9958053, 6867766
# POTENTIAL FOR TEX: 9517393, 6607447, 6137704
# RESULTS:  -acc=98% 
#          
#          
# FINAL CUT: 10004738 92.8747 145.935 0.3100833333333333   acc=91.66%
#            10004738 56.4757 143.812 0.25233333333333335  acc=95.64%
#            8292840 100.283 144.762 0.44916666666666666   acc=93.84%
#            11391018 30.3604 148.091 0.252                acc=77% old 73% new
#            3656121 31.1588 142.744 0.21820833333333334   acc=98% old 95.63% new

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


