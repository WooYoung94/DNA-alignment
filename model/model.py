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

class seqMLP(nn.Module):

	def __init__(self):

		super(seqMLP, self).__init__()

		self.enc = Encoder(32 * 5, 128, norm = 'none')
		self.fc1 = LinearBlock(128, 128, norm = 'none', activation = 'relu')
		self.fc2 = LinearBlock(128, 128, norm = 'none', activation = 'relu')
		self.fc3 = LinearBlock(128, 128, norm = 'none', activation = 'relu')
		self.fc4 = LinearBlock(128, 128, norm = 'none', activation = 'relu')
		self.fc5 = LinearBlock(128, 1, norm = 'none', activation = 'relu')

		layers = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]
		self.main = nn.Sequential(*layers)

	def forward(self, seq):

		seq = seq.view(seq.size(0), -1).float()

		out = self.enc(seq)
		out = self.main(out)

		return out

class seqGRU(nn.Module):

	def __init__(self):

		super(seqGRU, self).__init__()
		
		self.gru = nn.GRU(4, 256, 2, batch_first = True, bidirectional = True, dropout = 0.1)
		self.fc1 = nn.Linear(512, 64)
		self.fc2 = nn.Linear(64, 1)
		self.sigmoid = nn.Sigmoid()
		self.relu = nn.ReLU(inplace = True)

	def forward(self, seq):

		seq = torch.transpose(seq, 1, 2).float()
		h0 = torch.zeros(4, seq.size(0), 256).cuda()

		out, _ = self.gru(seq, h0)
		out = self.fc1(out[:, -1, :])
		out = self.relu(out)
		out = self.fc2(out)
		out = self.sigmoid(out) * 0.6 + 0.2

		return out

class resBlock1D(nn.Module):

	def __init__(self, inDim, outDim, size = 3, stride = 1):

		super(resBlock1D, self).__init__()

		self.inDim = inDim
		self.outDim = outDim

		self.conv0 = nn.Conv1d(inDim, outDim, size, stride = stride, padding = 1, bias = False)
		self.bn0 = nn.BatchNorm1d(outDim)
		self.conv1 = nn.Conv1d(outDim, outDim, size, stride = stride, padding = 1, bias = False)
		self.bn1 = nn.BatchNorm1d(outDim)
		self.conv2 = nn.Conv1d(outDim, outDim, size, stride = stride, padding = 1, bias = False)
		self.bn2 = nn.BatchNorm1d(outDim)
		self.relu = nn.ReLU(inplace = True)

	def forward(self, x):

		if self.inDim != self.outDim:

			x = self.conv0(x)
			x = self.bn0(x)

		print(out.shape)
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)

		out = out + x
		out = self.relu(out)

		return out

class seqCNN(nn.Module):

	def __init__(self):

		super(seqCNN, self).__init__()

		self.enc = nn.Conv2d(1, 64, (4, 3), stride = 1, padding = (0, 1))
		self.res1 = resBlock1D(64, 128, 3)
		self.res2 = resBlock1D(128, 256, 3)
		self.res3 = resBlock1D(256, 512, 32)
		self.conv1 = nn.Conv1d(512, 1, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, seq):

		#seq = torch.transpose(seq, 1, 2).float().view(seq.size(0), 1, seq.size())
		seq = seq.float().view(seq.size(0), 1, seq.size(1), seq.size(2))
		
		out = self.enc(seq)
		out = out.view(out.size(0), out.size(1), out.size(3))
		out = self.res1(out)
		out = self.res2(out)
		out = self.res3(out)
		out = self.conv1(out)
		out = self.sigmoid(out) * 0.6 + 0.2

		return out