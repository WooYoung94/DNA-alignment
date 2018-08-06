import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import model as m
import dataLoader as dl

def test(seqModel):

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	batchSize = 1
	shuffle = True

	seqData = dl.sequenceDataset('/home/dataset/genome/hg38/devData/testTrain_chrM.fa')
	seqDataLoader = torch.utils.data.DataLoader(dataset = seqData, batch_size = batchSize, shuffle = shuffle)

	criterion = nn.MSELoss()

	seqModel = seqModel.to(device)
	seqModel.eval()
	
	for idx, (s1, s2, y) in enumerate(seqDataLoader):

		s11 = s1.to(device)
		s22 = s2.to(device)
		yy = y.to(device)

		out = seqModel(s11, s22)
		loss = criterion(out, yy)

		o = out.detach().numpy()

		print('True : [{}/{}], Pred : [{}/{}], Delta : [{}/{}], Loss : {}'.format(y[0], y[1], o[0], o[1], y[0] - o[0], y[1] - o[1], loss.item()))