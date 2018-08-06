import os
import sys
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F

import model as m
import dataLoader as dl
import train






if __name__ == '__main__':

	seqModel = m.seqMLP()

	timeNow = timeit.default_timer()
	print('[info] train start')
	train.train(seqModel)
	trainTime = timeit.default_timer() - timeNow
	print('[info] elapsed time : ', trainTime)

	timeNow = timeit.default_timer()
	print('[info] test start')
	test.test(seqModel)
	testTime = timeit.default_timer() - timeNow
	print('[info] elapsed time : ', testTime)