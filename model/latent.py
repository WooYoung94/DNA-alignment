import os
import sys
import timeit
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

import model as m
import dataLoader as dl
import train
import test

if __name__ == '__main__':

	seqModel = m.seqCNN()

	print('[info] load {}'.format(sys.argv[1]))
	seqModel.load_state_dict(torch.load(sys.argv[1]))

	timeNow = timeit.default_timer()
	print('[info] test start')
	latent = test.testLatent(seqModel, batchSize = 1, dataset = sys.argv[2])
	testTime = timeit.default_timer() - timeNow
	print('[info] elapsed time : ', testTime)

	with open(sys.argv[3], 'wb') as fs:
	
		pickle.dump(latent, fs)