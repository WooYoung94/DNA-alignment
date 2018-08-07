import os
import sys
import numpy as np
from torch.utils.data import Dataset

def splitLine(l):

	return l.split('\t')

def seq2vec(s):

	s = s.upper()
	s = s.replace('A', '1')
	s = s.replace('G', '2')
	s = s.replace('T', '3')
	s = s.replace('C', '4')
	s = s.replace('N', '5')

	return np.array(list(s), dtype = np.float32)

class sequenceDataset(Dataset):

	def __init__(self, file):

		with open(file, 'r') as fs:

			lines = fs.readlines()[:-1]
 
		self.data = list(map(splitLine, lines))

	def __len__(self):

		return len(self.data)

	def __getitem__(self, idx):

		item = self.data[idx]

		return seq2vec(item[0]), seq2vec(item[1]), np.array(list(item[2]), dtype = np.float32), np.array([item[3], item[4]], dtype = np.float32)