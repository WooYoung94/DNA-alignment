import torch
import torch.nn as nn
import torch.nn.functional as F

class seqMLP(nn.Module):

	def __init__(self):

		super(seqMLP, self).__init__()

		self.s1fc1 = nn.Linear(128, 256)
		self.s1fc2 = nn.Linear(256, 256)
		self.s1fc3 = nn.Linear(256, 256)
		
		self.s2fc1 = nn.Linear(128, 256)
		self.s2fc2 = nn.Linear(256, 256)
		self.s2fc3 = nn.Linear(256, 256)

		self.fc4 = nn.Linear(513, 256)
		self.fc5 = nn.Linear(256, 256)
		self.fc6 = nn.Linear(256, 2)

	def forward(self, seq1, seq2, gap):

		gap = gap.reshape(-1, 1)

		out1 = seq1
		out1 = self.s1fc1(out1)
		out1 = F.relu(out1, inplace = True)
		out1 = self.s1fc2(out1)
		out1 = F.relu(out1, inplace = True)
		out1 = self.s1fc3(out1)
		out1 = F.relu(out1, inplace = True)

		out2 = seq2
		out2 = self.s1fc1(out2)
		out2 = F.relu(out2, inplace = True)
		out2 = self.s1fc2(out2)
		out2 = F.relu(out2, inplace = True)
		out2 = self.s1fc3(out2)
		out2 = F.relu(out2, inplace = True)	

		out = torch.cat((out1, out2, gap), dim = 1)
		out = self.fc4(out)
		out = F.relu(out, inplace = True)	
		out = self.fc5(out)
		out = F.relu(out, inplace = True)	
		out = self.fc6(out)

		return out

class seqGRU(nn.Module):

	def __init__(seqGRU, self).__init__():

		pass

	def forward(self, seq1, seq2, gap):

		pass