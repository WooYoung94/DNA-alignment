import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
	
	def __init__(self, num_features, eps=1e-5, affine=True):
		
		super(LayerNorm, self).__init__()
		
		self.num_features = num_features
		self.affine = affine
		self.eps = eps

		if self.affine:
		
			self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
			self.beta = nn.Parameter(torch.zeros(num_features))

	def forward(self, x):
		
		shape = [-1] + [1] * (x.dim() - 1)
		mean = x.view(x.size(0), -1).mean(dim=1).view(*shape)
		std = x.view(x.size(0), -1).std(dim=1).view(*shape)
		x = (x - mean) / (std + self.eps)

		if self.affine:
			
			shape = [1, -1] + [1] * (x.dim() - 2)
			x = x * self.gamma.view(*shape) + self.beta.view(*shape)
		
		return x

class LinearBlock(nn.Module):

	def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
		
		super(LinearBlock, self).__init__()
	
		if norm == 'bn':
			
			self.norm = nn.BatchNorm1d(output_dim)
		
		elif norm == 'ln':
			
			self.norm = LayerNorm(output_dim)

		elif norm == 'none':

			self.norm = None

		if activation == 'relu':
			
			self.activation = nn.ReLU(inplace=True)
		
		elif activation == 'none':
			
			self.activation = None

		layers = list()

		layers.append(nn.Linear(input_dim, output_dim))
		self.main = nn.Sequential(*layers)

	def forward(self, x):
		
		x = self.main(x)
		
		if self.norm:
			
			x = self.norm(x)
		
		if self.activation:
			
			x = self.activation(x)
		
		return x

class Encoder(nn.Module):

	def __init__(self, input_dim, output_dim, repeat_num = 2, norm = 'none', activation = 'relu'):

		super(Encoder, self).__init__()
		
		layers = list()
		curr_dim = input_dim
		
		for _ in range(repeat_num):

			layers.append(LinearBlock(curr_dim, curr_dim // 2, norm = norm, activation = activation))
			curr_dim = curr_dim // 2

		layers.append(LinearBlock(curr_dim, output_dim, norm = norm, activation = activation))
		self.main = nn.Sequential(*layers)

	def forward(self, x):

		return self.main(x)

class seqMLP(nn.Module):

	def __init__(self):

		super(seqMLP, self).__init__()

		self.enc = Encoder(128 * 5, 128, norm = 'none')

		self.s1fc1 = LinearBlock(128, 256, norm = 'none')
		self.s1fc2 = LinearBlock(256, 256, norm = 'none')
		self.s1fc3 = LinearBlock(256, 256, norm = 'none')
		
		self.s2fc1 = LinearBlock(128, 256, norm = 'none')
		self.s2fc2 = LinearBlock(256, 256, norm = 'none')
		self.s2fc3 = LinearBlock(256, 256, norm = 'none')

		self.fc4 = LinearBlock(513, 128, norm = 'none')
		self.fc5 = LinearBlock(128, 128, norm = 'none')
		self.fc6 = LinearBlock(128, 128, norm = 'none')

	def forward(self, seq1, seq2, gap):

		gap = gap.reshape(-1, 1)
		seq1 = seq1.view(seq1.size(0), -1).float()
		seq2 = seq2.view(seq2.size(0), -1).float()

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