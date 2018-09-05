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

def main():

	log = ['=']
	#seqModel = m.seqGRU()
	seqModel = m.seqCNN()

	timeNow = timeit.default_timer()
	print('[info] train start')
	log = train.train(seqModel) + log
	trainTime = timeit.default_timer() - timeNow
	print('[info] elapsed time : ', trainTime)

	timeNow = timeit.default_timer()
	print('[info] test start')
	log = log + test.test(seqModel, batchSize = 1)
	testTime = timeit.default_timer() - timeNow
	print('[info] elapsed time : ', testTime)

	timeNow = timeit.default_timer()
	print('[info] inference test start')
	test.test(seqModel, batchSize = 1024)
	testTime = timeit.default_timer() - timeNow
	print('[info] elapsed time : ', testTime)

	with open(sys.argv[1], 'w') as fs:

		fs.write(''.join(log))

if __name__ == '__main__':

	main()