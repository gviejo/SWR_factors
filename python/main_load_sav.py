from lfads import LFADS
import numpy as np
import os
import tensorflow as tf
import re
import sys
import h5py
from functions import *
from time import time
from lfads.utils import *
from lfads.functions import *
import pickle
from sklearn.model_selection import train_test_split



# data_directory = '../data/lfads_test_colab/'
data_directory = '/mnt/DataGuillaume/Factors/model_runs_Mouse12'

files = os.listdir(data_directory)

samples = {}

for f in files:
	k = f.split("_")[0]	
	samples[k] = read_data(os.path.join(data_directory, f))


datasets = pickle.load(open("../data/swr_hist_Mouse12.pickle", "rb"))
datasets = {k.split("/")[1]:datasets[k] for k in datasets.keys()}


def ploti(s,k=0):
	n = datasets[s]['train_data'].shape[-1]
	for i in range(n):
		plot(datasets[s]['train_data'][k,:,i]+i*2, color = 'black')
		plot(samples[s]['output_dist_params'][k,:,i]+i*2, color = 'red')
		axvline(100, alpha = 0.5)

sys.exit()



factors = []

for s in samples.keys():
	dims = samples[s]['factors'].shape
	factors.append(samples[s]['factors'].reshape(dims[0], dims[1] * dims[2]))
	# factors.append(samples[s]['train']['factors'][:,0,:])

factors = np.vstack(factors)

from sklearn.manifold import TSNE

X = TSNE(n_components=2, perplexity = 30).fit_transform(factors)


from pylab import *

scatter(X[:,0], X[:,1])

show()


