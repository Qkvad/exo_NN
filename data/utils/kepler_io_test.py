#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 18:23:42 2018

@author: sandro
"""

import kepler_io as kepio

import matplotlib.pyplot as plt
import numpy as np

import sys


KEP_DIR = "data/kepler/" #"../test_data"
KEP_ID   = 11442793

def show_lightcurve(kep_id):

	filenames = kepio.get_filenames(KEP_DIR, kep_id)
	all_time, all_flux = kepio.read_light_curve(filenames)
		
	for f in all_flux:
	    f /= np.median(f)

	all_time_concat = np.concatenate(all_time)
	all_flux_concat = np.concatenate(all_flux)
	q3max = all_flux[3].max()
	
	plt.figure(figsize=(12, 6), dpi=80)
	plt.subplot(211)    
	plt.scatter(all_time_concat, all_flux_concat, c=[(value if value<q3max else 1.0) for value in all_flux_concat], s=0.5, cmap='plasma')
	plt.title('Entire Kepler Mission Data for KEP_ID=' + str(kep_id))
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

def show_processed_lightcurve(kep_id):

	all_time, all_flux = preprocess.read_and_process_light_curve(kep_id, KEP_DIR, max_gap_width=0.75)

	#all_time_concat = np.concatenate(all_time)
	#all_flux_concat = np.concatenate(all_flux)
	q3max = all_flux.max()
	
	plt.figure(figsize=(12, 6), dpi=80)    
	plt.scatter(all_time, all_flux, c=[(value if value<q3max else 1.0) for value in all_flux], s=0.5, cmap='plasma')
	plt.title('Entire Kepler Mission Data for KEP_ID=' + str(kep_id))
	plt.xlabel('Time(days)')
	plt.ylabel('Brightness')
	plt.show()
#show_lightcurve(KEP_ID)

if __name__=='__main__':
  show_lightcurve(KEP_ID)
