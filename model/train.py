import os
import sys
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import scipy as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import model as m
import dataLoader as dl

def train():

	modelPath = '/home/youngwoo/Documents/models/DNA/model_epoch{}.model'
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	batchSize = 128
	shuffle = True
	learningRate = 0.001
	epochNum = 100

	seqData = dataLodaer.sequenceDataset('/home/dataset/genome/hg38/devData/testTrain_chrM.fa')
	seqDataLoader = torch.utils.data.DataLoader(dataset = seqData, batch_size = batchSize, shuffle = shuffle)

	seqModel = m.seqMLP()
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(list(seqModel.parameters()), lr = learningRate)
	totalStep = len(seqDataLoader)

	seqModel.train()

	for epoch in range(epochNum):

		for idx, (s1, s2, y1, y2) in enumerate(seqDataLoader):

			s1 = s1.to(device)
			s2 = s2.to(device)
			y1 = y1.to(device)
			y2 = y2.to(device)
			y = torch.cat((y1, y2), dim = 1)

			out = seqModel(s1, s2)
			loss = criterion(out, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if (idx + 1) % 100 == 0:

				print('Epoch : [{}/{}], Step : [{}/{}], Loss : {}'.format(epoch + 1, epochNum, idx + 1, totalStep, loss.item()))

		if epoch % 10 == 0 and epoch > 0:

			torch.save(seqModel.state_dict(), modelPath.format(epoch))