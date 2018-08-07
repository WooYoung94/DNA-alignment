import torch
import numpy as np

def get_onehot(labels, dim=5):
	batch_size = labels.size(0)
	out = torch.zeros(batch_size, dim)
	out[np.arange(batch_size), labels.long()] = 1
	return out