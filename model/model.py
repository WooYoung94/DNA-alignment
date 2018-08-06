import torch
import torch.nn as nn
import torch.nn.functional as F

class seqMLP(nn.Module):

	def __init__(self):

		super(seqMLP, self).__init__()

		self.fc1 = nn.Linear(256, 128)
		self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(128, 2)

	def forward(self, s1, s2):

		out = torch.cat((s1, s2), dim = 1)
		out = self.fc1(out)
		out = F.relu(out, inplace = True)
		out = self.fc2(out)
		out = F.relu(out, inplace = True)	
		out = self.fc3(out)

		return out