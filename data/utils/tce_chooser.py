#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 18:20:03 2018

@author: sandro
"""

# PC  - 1058
# AFP - 471
# NTP - 734

import sys

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

np.random.seed(427)

def reduce_table(pc_num, afp_num, ntp_num, output_csv):
    df = pd.read_csv("data/csv/dr24_tce.csv", index_col="rowid", comment="#")
    #df = df[df["kepid"] != 11442793]

    print('df shape: ', df.shape)

    df_PC = df[df["av_training_set"] == "PC"]
    print("total PCs :", df_PC.shape[0])
    df_AFP = df[df["av_training_set"] == "AFP"]
    print("total AFPs:", df_AFP.shape[0])
    df_NTP = df[df["av_training_set"] == "NTP"]
    print("total NTPs:", df_NTP.shape[0])
    df_UNK = df[df["av_training_set"] == "UNK"]
    print("total UNKs:", df_UNK.shape[0])

    '''# remove astrophisical false positives
    df = df[df["av_training_set"] != "AFP"]
    print("\n")

    df_PC = df[df["av_training_set"] == "PC"]
    print("total PCs :", df_PC.shape[0])
    df_AFP = df[df["av_training_set"] == "AFP"]
    print("total AFPs:", df_AFP.shape[0])
    df_NTP = df[df["av_training_set"] == "NTP"]
    print("total NTPs:", df_NTP.shape[0])'''

    pc_rm_num = df_PC.shape[0] - pc_num;
    if pc_rm_num < 0: 
        pc_rm_num = 0
    drop_pc_indices = np.random.choice(df_PC.index, pc_rm_num, replace=False)
    pc_subset = df_PC.drop(drop_pc_indices)
    ######
    afp_rm_num = df_AFP.shape[0] - afp_num;
    if afp_rm_num < 0: 
        afp_rm_num = 0
    drop_afp_indices = np.random.choice(df_AFP.index, afp_rm_num, replace=False)
    afp_subset = df_AFP.drop(drop_afp_indices)
    ######
    ntp_rm_num = df_NTP.shape[0] - ntp_num;
    if ntp_rm_num < 0: 
        ntp_rm_num = 0
    drop_ntp_indices = np.random.choice(df_NTP.index, ntp_rm_num, replace=False)
    ntp_subset = df_NTP.drop(drop_ntp_indices)

    frames = [pc_subset, afp_subset, ntp_subset]
    df_subset = pd.concat(frames)
    df_subset = shuffle(df_subset)
    
    print("\n")
    print("planet candidates         : ", df_subset.loc[df_subset["av_training_set"] == "PC"].shape[0])
    print("astrophy. false positives : ", df_subset.loc[df_subset["av_training_set"] == "AFP"].shape[0])
    print("non-transiting phenomena  : ", df_subset.loc[df_subset["av_training_set"] == "NTP"].shape[0])

    df_subset.to_csv("data/csv/"+output_csv+".csv", sep=',')

# argv parameters:
#  - (int) argv[1]: number of PCs
#  - (int) argv[2]: number of AFPs
#  - (int) argv[3]: number of NTPs
#  - (str) argv[4]: output csv file name (without .csv)
if __name__=='__main__':
  reduce_table(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),str(sys.argv[4]))


