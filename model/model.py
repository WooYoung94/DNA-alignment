import torch
import torch.nn as nn
import torch.nn.functional as F

class seqMLP(nn.Module):

	def __init__(self):

		super(seqMLP, self).__init__()

		self.fc1 = nn.Linear(257, 512)
		self.fc2 = nn.Linear(512, 512)
		self.fc3 = nn.Linear(512, 512)
		self.fc4 = nn.Linear(512, 2)

	def forward(self, s1, s2, gap):

		print(s1.shape)
		print(s2.shape)
		print(gap.shape)

		out = torch.cat((s1, s2, gap), dim = 1)
		out = self.fc1(out)
		out = F.relu(out, inplace = True)
		out = self.fc2(out)
		out = F.relu(out, inplace = True)	
		out = self.fc3(out)
		out = F.relu(out, inplace = True)	
		out = self.fc4(out)

		return out