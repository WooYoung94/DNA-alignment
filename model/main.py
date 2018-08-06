import os
import sys
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F

import model as m
import dataLoader as dl
import train as t






if __name__ == '__main__':

	timeNow = timeit.default_timer()
	print('[info] train start')
	t.train()
	trainTime = timeit.default_timer() - timeNow
	print('[info] elapsed time : ', trainTime)