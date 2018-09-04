import os
import sys
import numpy as np
from torch.utils.data import Dataset

SEQLENGTH = 32

def splitLine(l):

	return l.split('\t')

def seq2vec(s, onehot = True):

	s = s.upper()
	s = s.replace('A', '0')
	s = s.replace('G', '1')
	s = s.replace('T', '2')
	s = s.replace('C', '3')
	s = s.replace('W', '4')
	s = list(s)

	if onehot:

		vec = np.zeros((5, len(s)))
		vec[np.array(s, dtype = np.int32), np.arange(len(s))] = 1

		return np.array(vec, dtype = np.int32)

	else:

		return s

class sequenceDataset(Dataset):

	def __init__(self, file):

		with open(file, 'r') as fs:

			lines = fs.readlines()[:-1]
 
		self.data = list(map(splitLine, lines))

	def __len__(self):

		return len(self.data)

	def __getitem__(self, idx):

		item = self.data[idx]

		return seq2vec(item[0]), np.array([item[1]], dtype = np.float32)


"""
def getSeqence(seq, idx, length):

	return seq[idx:idx + length]

class sequenceDataset(Dataset):

	def __init__(self, file):

		with open(file, 'r') as fs:

			self.seq = fs.read()[:-1]


		self.dataIdx = [idx for idx, val in enumerate(self.seq) if val != 'N']

	def __len__(self):

		return len(self.dataIdx) - SEQLENGTH + 1

	def __getitem__(self, idx):

		return seq2vec(getSeqence(self.seq), self.dataIdx[idx], SEQLENGTH), np.array([self.dataIdx[idx]], dtype = np.float32)
"""