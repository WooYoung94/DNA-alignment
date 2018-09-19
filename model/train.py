import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import model as m
import dataLoader as dl

MAX_LENGTH = 16505

def KLD(mu, logvar):

	return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def train(seqModel, param = None):

	log = list()

	if param:

		pass

	else:

		modelPath = '/home/youngwoo/Documents/models/DNA/cnn2.model'
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		batchSize = 1024
		shuffle = True
		learningRate = 0.0001
		epochNum = 2000

		seqData = dl.sequenceDataset('/home/dataset/genome/hg38/devData/trainData_chrM_augmented(origin+3).fa')
		seqDataLoader = torch.utils.data.DataLoader(dataset = seqData, batch_size = batchSize, shuffle = shuffle)

		seqModel = seqModel.to(device)
		criterion = nn.MSELoss()
		optimizer = torch.optim.Adam(list(seqModel.parameters()), lr = learningRate)
		totalStep = len(seqDataLoader)

	seqModel.train()

	for epoch in range(epochNum):

		for idx, (s, y) in enumerate(seqDataLoader):

			s = s.to(device)
			y = (y / MAX_LENGTH).to(device)

			out = seqModel(s)
			loss = criterion(out, y)
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print('Epoch : [{}/{}], Loss : {}'.format(epoch + 1, epochNum, loss.item()))
		log.append('Epoch : [{}/{}], Loss : {}\n'.format(epoch + 1, epochNum, loss.item()))

	torch.save(seqModel.state_dict(), modelPath)

	return log