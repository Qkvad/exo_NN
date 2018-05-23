#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 18:23:42 2018

@author: sandro
"""

import kepler_io as kepio

BASE_DIR = "test_data/"
KEP_ID   = 11442793

DO_PLOT  = True

filenames = kepio.get_filenames(BASE_DIR, KEP_ID)
all_time, all_flux = kepio.read_light_curve(filenames)

if not DO_PLOT:
    print("all_time length:", len(all_time))
    print("all_flux length:", len(all_flux))

else:
    import matplotlib.pyplot as plt
    import numpy as np
    
    for f in all_flux:
        f /= np.median(f)
        
    plt.plot(np.concatenate(all_time), np.concatenate(all_flux), ".")
    plt.show()