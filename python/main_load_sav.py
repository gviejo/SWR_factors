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



data_directory = '../data/lfads_test_colab/'

files = os.listdir(data_directory)

samples = {}

for f in files:
	k = f.split("_")[0]
	if k not in samples:
		samples[k] = {'train':{}, 'valid':{}}
	if 'train' in f:
		samples[k]['train'] = read_data(os.path.join(data_directory, f))
	elif 'valid' in f:
		samples[k]['valid'] = read_data(os.path.join(data_directory, f))



sys.exit()

factors = []

for s in samples.keys():
	dims = samples[s]['train']['factors'].shape
	factors.append(samples[s]['train']['factors'].reshape(dims[0], dims[1] * dims[2]))
	# factors.append(samples[s]['train']['factors'][:,0,:])

factors = np.vstack(factors)

from sklearn.manifold import TSNE

X = TSNE(n_components=2, perplexity = 30).fit_transform(factors)

scatter(X[:,0], X[:,1])


