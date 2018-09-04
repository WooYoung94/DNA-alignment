import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import model as m
import dataLoader as dl

MAX_LENGTH = 16505

def test(seqModel, batchSize = 1):

	log = list()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	shuffle = True

	seqData = dl.sequenceDataset('/home/dataset/genome/hg38/devData/trainData_chrM_original.fa')
	seqDataLoader = torch.utils.data.DataLoader(dataset = seqData, batch_size = batchSize, shuffle = shuffle)

	criterion = nn.MSELoss()

	seqModel = seqModel.to(device)
	seqModel.eval()
	
	for idx, (s, y) in enumerate(seqDataLoader):

		s = s.to(device)
		y = y.to(device)

		out, mu, logvar = seqModel(s)
		out = out * MAX_LENGTH
		#out = seqModel(s) * MAX_LENGTH
		loss = criterion(out, y)

		y = y.cpu().numpy()[0].astype(np.int32)
		o = out.detach().cpu().numpy()[0].astype(np.int32)

		if batchSize == 1:

			print('True/Pred : [{}/{}], Loss : {}'
				.format(y, o, loss.item()))
			log.append('True/Pred : [{}/{}], Loss : {}\n'
				.format(y, o, loss.item()))

	return log

def testVariant(seqModel, batchSize = 1):

	log = list()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	shuffle = True

	seqData = dl.sequenceDataset('/home/dataset/genome/hg38/devData/trainData_chrM_variant_2.fa')
	seqDataLoader = torch.utils.data.DataLoader(dataset = seqData, batch_size = batchSize, shuffle = shuffle)

	criterion = nn.MSELoss()

	seqModel = seqModel.to(device)
	seqModel.eval()
	
	for idx, (s, y) in enumerate(seqDataLoader):

		s = s.to(device)
		y = y.to(device)

		#out, mu, logvar = seqModel(s)
		#out = out * MAX_LENGTH
		out = seqModel(s) * MAX_LENGTH
		loss = criterion(out, y)

		y = y.cpu().numpy()[0].astype(np.int32)
		o = out.detach().cpu().numpy()[0].astype(np.int32)

		if batchSize == 1:

			print('True/Pred : [{}/{}], Loss : {}'
				.format(y, o, loss.item()))
			log.append('True/Pred : [{}/{}], Loss : {}\n'
				.format(y, o, loss.item()))

	return log