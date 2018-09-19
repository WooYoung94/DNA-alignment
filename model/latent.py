import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

import model
import dataLoader as dl
import train
import test

if __name__ == '__main__':

	seqModel = m.seqCNN()

	print('[info] load {}'.format(sys.argv[1]))
	seqModel.load_state_dict(torch.load(sys.argv[1]))

	timeNow = timeit.default_timer()
	print('[info] test start')
	latent = test.testLatent(seqModel, batchSize = 1)
	testTime = timeit.default_timer() - timeNow
	print('[info] elapsed time : ', testTime)

	#with open(sys.argv[2], 'wb') as fs:
	#
	#	pickle.dump(latent, fs)