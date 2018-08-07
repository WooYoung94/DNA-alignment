import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearBlock(nn.Module):

	def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
		super(LinearBlock, self).__init__()
		layers = []

		if norm == 'bn':
			self.norm = nn.BatchNorm1d(output_dim)
		elif norm=='none':
			self.norm = None

		if activation == 'relu':
			self.activation = nn.ReLU(inplace=True)
		elif norm=='none':
			self.activation = None

		layers += [nn.Linear(input_dim, output_dim)]
		self.main = nn.Sequential(*layers)

	def forward(self, x):
		x = self.main(x)
		if self.norm:
			x = self.norm(x)
		if self.activation:
			x = self.activation(x)
		return x


class seqMLP(nn.Module):

	def __init__(self):

		super(seqMLP, self).__init__()

		self.enc = LinearBlock(128*5, 128, norm='bn')

		self.s1fc1 = LinearBlock(128, 256, norm='bn')
		self.s1fc2 = LinearBlock(256, 256, norm='bn')
		self.s1fc3 = LinearBlock(256, 256, norm='bn')
		
		self.s2fc1 = LinearBlock(128, 256, norm='bn')
		self.s2fc2 = LinearBlock(256, 256, norm='bn')
		self.s2fc3 = LinearBlock(256, 256, norm='bn')

		self.fc4 = LinearBlock(513, 256, norm='bn')
		self.fc5 = LinearBlock(256, 256, norm='bn')
		self.fc6 = LinearBlock(256, 2, norm='bn')

	def forward(self, seq1, seq2, gap):

		gap = gap.reshape(-1, 1)

		out1 = self.enc(seq1)
		out2 = self.enc(seq2)

		# out1 = seq1
		out1 = self.s1fc1(out1)
		out1 = self.s1fc2(out1)
		out1 = self.s1fc3(out1)

		# out2 = seq2
		out2 = self.s1fc1(out2)
		out2 = self.s1fc2(out2)
		out2 = self.s1fc3(out2)	

		out = torch.cat((out1, out2, gap), dim = 1)
		out = self.fc4(out)
		out = self.fc5(out)
		out = self.fc6(out)

		return out
"""
class seqGRU(nn.Module):

	def __init__(seqGRU, self).__init__():

		super(seqGRU, self).__init__()
		pass

	def forward(self, seq1, seq2, gap):

		pass
"""