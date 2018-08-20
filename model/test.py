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

	seqData = dl.sequenceDataset('/home/dataset/genome/hg38/devData/test_chrM.fa')
	seqDataLoader = torch.utils.data.DataLoader(dataset = seqData, batch_size = batchSize, shuffle = shuffle)

	criterion = nn.MSELoss()

	seqModel = seqModel.to(device)
	seqModel.eval()
	
	for idx, (s, y) in enumerate(seqDataLoader):

		s = s.to(device)
		#y = y.to(device)

		out = seqModel(s)
		loss = criterion(out, y)

		y = y.numpy()[0].astype(np.int32)
		o = out.detach().cpu().numpy()[0].astype(np.int32)


		print('True/Pred : [{}/{}], Loss : {}'
			.format(y, out, loss.item()))
		log.append('True/Pred : [{}/{}], Loss : {}\n'
			.format(y, out, loss.item()))

	return log