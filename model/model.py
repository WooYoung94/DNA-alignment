import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
	
	def __init__(self, num_features, eps = 1e-5, affine = True):
		
		super(LayerNorm, self).__init__()
		
		self.num_features = num_features
		self.affine = affine
		self.eps = eps

		if self.affine:
		
			self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
			self.beta = nn.Parameter(torch.zeros(num_features))

	def forward(self, x):
		
		shape = [-1] + [1] * (x.dim() - 1)
		mean = x.view(x.size(0), -1).mean(dim = 1).view(*shape)
		std = x.view(x.size(0), -1).std(dim = 1).view(*shape)
		x = (x - mean) / (std + self.eps)

		if self.affine:
			
			shape = [1, -1] + [1] * (x.dim() - 2)
			x = x * self.gamma.view(*shape) + self.beta.view(*shape)
		
		return x

class LinearBlock(nn.Module):

	def __init__(self, input_dim, output_dim, norm = 'none', activation = 'relu'):
		
		super(LinearBlock, self).__init__()
	
		if norm == 'bn':
			
			self.norm = nn.BatchNorm1d(output_dim)
		
		elif norm == 'ln':
			
			self.norm = LayerNorm(output_dim)

		elif norm == 'none':

			self.norm = None

		if activation == 'relu':
			
			self.activation = nn.ReLU(inplace = True)
		
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

class seqGRU(nn.Module):

	def __init__(self):

		super(seqGRU, self).__init__()
		
		self.gru = nn.GRU(5, 256, 2, batch_first = True, bidirectional = True, dropout = 0.1)
		self.fc1 = nn.Linear(512, 64)
		self.fc2 = nn.Linear(64, 1)
		self.fc1_drop = nn.Dropout(0.1)
		self.sigmoid = nn.Sigmoid()
		self.relu = nn.ReLU(inplace = True)

	def forward(self, seq):

		seq = torch.transpose(seq, 1, 2).float()
		h0 = torch.zeros(4, seq.size(0), 256).cuda()

		out, _ = self.gru(seq, h0)
		out = self.fc1(out[:, -1, :])
		out = self.relu(out)
		out = self.fc1_drop(out)
		out = self.fc2(out)
		out = self.sigmoid(out)

		return out


class seqCNN(nn.Module):

	def __init__(self):

		super(seqCNN, self).__init__()

		self.enc = nn.Conv2d(1, 64, (5, 3), stride = 1, padding = (0, 1))

		self.conv1 = nn.Conv1d(64, 128, 3, padding = 1)
		self.conv2 = nn.Conv1d(128, 256, 3, padding = 1)
		self.conv3 = nn.Conv1d(256, 512, 3, padding = 1)
		self.conv4 = nn.Conv1d(512, 1024, 3, padding = 1)
		self.conv5 = nn.Conv1d(1024, 1, 32)
		
		self.relu = nn.ReLU(inplace = True)
		self.sigmoid = nn.Sigmoid()

	def reparam(self, x):

		mu = x[:, :1024, :]
		logvar = x[:, 1024:, :]
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)

		return eps.mul(std).add_(mu), mu, logvar

	def forward(self, seq):

		#seq = torch.transpose(seq, 1, 2).float().view(seq.size(0), 1, seq.size())
		seq = seq.float().view(seq.size(0), 1, seq.size(1), seq.size(2))
		
		out = self.enc(seq)
		out = out.view(out.size(0), out.size(1), out.size(3))
		
		out = self.relu(self.conv1(out))
		out = self.relu(self.conv2(out))
		out = self.relu(self.conv3(out))
		out = self.conv4(out)

		#out, mu, logvar = self.reparam(out)

		out = self.conv5(out)
		out = self.sigmoid(out)
		out = out.view(out.size(0), out.size(1))

		return out


"""
class seqCNN(nn.Module):

	def __init__(self):

		super(seqCNN, self).__init__()

		self.enc = nn.Conv2d(1, 64, (5, 3), stride = 1, padding = (0, 1))

		self.conv1_3 = nn.Conv1d(64, 128, 3, padding = 0)
		self.conv1_9 = nn.Conv1d(64, 256, 9, padding = 0)
		self.conv1_15 = nn.Conv1d(64, 512, 15, padding = 0)

		self.conv2_3 = nn.Conv1d(128, 512, 10, padding = 0)
		self.conv2_9 = nn.Conv1d(256, 512, 8, padding = 0)
		self.conv2_15 = nn.Conv1d(512, 1024, 6, padding = 0)

		self.fc1 = nn.Linear(2048, 128)
		self.fc2 = nn.Linear(128, 1)
	
		self.relu = nn.ReLU(inplace = True)
		self.drop = nn.Dropout(0.1)
		self.maxpool3 = nn.MaxPool1d(3)
		self.sigmoid = nn.Sigmoid()

	def forward(self, seq):

		seq = seq.float().view(seq.size(0), 1, seq.size(1), seq.size(2))
		
		out = self.enc(seq)
		out = out.view(out.size(0), out.size(1), out.size(3))
		
		out1_3 = self.maxpool3(self.relu(self.conv1_3(out)))
		out1_9 = self.maxpool3(self.relu(self.conv1_9(out)))
		out1_15 = self.maxpool3(self.relu(self.conv1_15(out)))

		#print('1_3', out1_3.shape)
		#print('1_9', out1_9.shape)
		#print('1_15', out1_15.shape)

		out2_3 = self.relu(self.conv2_3(out1_3))
		out2_9 = self.relu(self.conv2_9(out1_9))
		out2_15 = self.relu(self.conv2_15(out1_15))

		#print('2_3', out2_3.shape)
		#print('2_9', out2_9.shape)
		#print('2_15', out2_15.shape)

		out = torch.cat((out2_3, out2_9, out2_15), dim = 1)
		out = out.view(out.size(0), out.size(1))

		#print('out', out.shape)

		out = self.relu(self.fc1(out))
		out = self.drop(out)
		out = self.fc2(out)
		out = self.sigmoid(out)

		#print('out', out.shape)

		return out
"""