import os
import sys
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F

import model as m
import dataLoader as dl
import train
import test

if __name__ == '__main__':

	log = ['=']
	#seqModel = m.seqGRU()
	seqModel = m.seqCNN()

	print('[info] load {}'.format(sys.argv[1]))
	seqModel.load_state_dict(torch.load(sys.argv[1]))

	timeNow = timeit.default_timer()
	print('[info] test start')
	log = log + test.testVariant(seqModel, batchSize = 1)
	testTime = timeit.default_timer() - timeNow
	print('[info] elapsed time : ', testTime)

	timeNow = timeit.default_timer()
	print('[info] inference test start')
	test.test(seqModel, batchSize = 1024)
	testTime = timeit.default_timer() - timeNow
	print('[info] elapsed time : ', testTime)

	with open(sys.argv[2], 'w') as fs:

		fs.write(''.join(log))