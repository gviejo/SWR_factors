import _pickle as pickle
import sys, os

#####################################################################################
# load_datasets
#####################################################################################
datasets = pickle.load(open("../data/swr_spike_count_all.pickle", "rb"))

inputs = []

for k in datasets.keys():		
	for s in datasets[k].keys():
		inputs.append((k,s))

datasets = None

for p in inputs:
	os.system('python main_test_lfads.py '+str(p[0])+' '+str(p[1]))


