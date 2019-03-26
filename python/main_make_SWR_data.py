#!/usr/bin/env python
'''
    File name: main_make_SWR_data.py
    Author: Guillaume Viejo
    Date created: 21/03/2019
    Python Version: 3.5.2


'''

import numpy as np
import pandas as pd
import scipy.io
from functions import *
from pylab import *
import os, sys
import neuroseries as nts
from time import time
from numba import jit
import pickle



@jit(nopython=True)
def hist_swr(rip_times, spikes_list, bin_time):
	count 			= np.zeros((len(rip_times), len(bin_time)-1, len(spikes_list)))
	for i, t in enumerate(rip_times):
		tmp = bin_time + t
		for j, spk in enumerate(spikes_list):
			a, _ = np.histogram(spk, tmp)
			count[i, :, j] = a

	return count



data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
datatosave = {}

bin_size 		= 10 # ms
nb_bins 		= 200
times 			= np.arange(0, bin_size*(nb_bins+1), bin_size) - (nb_bins*bin_size)/2
bins 			= np.arange(times[0]-bin_size/2, times[-1]+bin_size/2+bin_size, bin_size)



for session in datasets:
	if 'Mouse12' in session:		
		generalinfo 	= scipy.io.loadmat(data_directory+session+'/Analysis/GeneralInfo.mat')
		shankStructure 	= loadShankStructure(generalinfo)
		if len(generalinfo['channelStructure'][0][0][1][0]) == 2:
			hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][1][0][0] - 1
		else:
			hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][0][0][0] - 1		
		spikes,shank	= loadSpikeData(data_directory+session+'/Analysis/SpikeData.mat', shankStructure['thalamus'])		
		wake_ep 		= loadEpoch(data_directory+session, 'wake')
		sleep_ep 		= loadEpoch(data_directory+session, 'sleep')
		sws_ep 			= loadEpoch(data_directory+session, 'sws')
		rem_ep 			= loadEpoch(data_directory+session, 'rem')
		sleep_ep 		= sleep_ep.merge_close_intervals(threshold=1.e3)		
		sws_ep 			= sleep_ep.intersect(sws_ep)	
		rem_ep 			= sleep_ep.intersect(rem_ep)
		rip_ep,rip_tsd 	= loadRipples(data_directory+session)
		rip_ep			= sws_ep.intersect(rip_ep)	
		rip_tsd 		= rip_tsd.restrict(sws_ep)
		
		spikes_raw  	= [spikes[n].index.values for n in spikes]

		bin_time 		= (bins*1e3).astype(np.int)

		t = time()
		count = hist_swr(rip_tsd.index.values, spikes_raw, bin_time)
		print(time() -t)

		datatosave[session] = {	'num_steps': len(times), 
								'dt': np.array(bin_size*1e-3),
								'train_data': count.astype(np.int32),
								'data_dim': len(spikes),
								'train_percentage':np.array(1)
								}



pickle.dump(datatosave, open("../data/swr_hist_Mouse12.pickle", "wb"))
