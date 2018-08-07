import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import model as m
import dataLoader as dl

def test(seqModel):

	log = list()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	batchSize = 1
	shuffle = True

	seqData = dl.sequenceDataset('/home/dataset/genome/hg38/devData/testTrain_chrM.fa')
	seqDataLoader = torch.utils.data.DataLoader(dataset = seqData, batch_size = batchSize, shuffle = shuffle)

	criterion = nn.MSELoss()

	seqModel = seqModel.to(device)
	seqModel.eval()
	
	for idx, (s1, s2, g, y) in enumerate(seqDataLoader):

		s11 = s1.to(device)
		s22 = s2.to(device)
		gg = g.to(device)
		yy = y.to(device)

		out = seqModel(s11, s22, gg)
		loss = criterion(out, yy) + criterion(torch.abs(out[:,0] - out[:,1]), gg)

		y = y.numpy()[0].astype(np.int32)
		g = g.numpy()[0].astype(np.int32)
		o = out.detach().cpu().numpy()[0].astype(np.int32)

		print('True : [{}/{}], Pred : [{}/{}], Delta : [{}/{}], Gap : [{}/{}], Loss : {}'
			.format(y[0], y[1], o[0], o[1], y[0] - o[0], y[1] - o[1], np.abs(o[0] - o[1]), g, loss.item()))
		log.append('True : [{}/{}], Pred : [{}/{}], Delta : [{}/{}], Gap : [{}/{}], Loss : {}\n'
			.format(y[0], y[1], o[0], o[1], y[0] - o[0], y[1] - o[1], np.abs(o[0] - o[1]), g, loss.item()))

	return log