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

    all_time_concat = np.concatenate(all_time)
    all_flux_concat = np.concatenate(all_flux)
    q3max = all_flux[3].max()
    
    plt.figure(1)
    plt.subplot(211)    
    plt.scatter(all_time_concat, all_flux_concat, c=[(value if value<q3max else 1.0) for value in all_flux_concat], s=0.5, cmap='plasma')
    plt.title('Entire Kepler Mission Data for KEP_ID=11442793')
    plt.xlabel('Time(days)')
    plt.ylabel('Brightness')
    plt.subplot(212)
    plt.scatter(all_time[3],all_flux[3],c=all_flux[3], s=2, cmap='plasma')
    plt.title('3$^{rd}$ quarter')
    plt.ylim(all_flux[3].min(),q3max)
    plt.xlabel('Time(days)')
    plt.ylabel('Brightness')
    plt.figure(1).tight_layout()
    plt.show()
