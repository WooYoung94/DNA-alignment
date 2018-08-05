import os
import sys
import random

SEQLENGTH = 128
MIN_PAIR = 300
MAX_PAIR = 5000

def getSeqence(seq, idx, length):

	return seq[idx:idx + length]

if __name__ == '__main__':

	data = list()

	with open('./chr13.fa', 'r') as fs:

		seq = fs.read()

	for idx in range(len(seq[:-SEQLENGTH]) + 1):

		data.append(getSeqence(seq, idx, SEQLENGTH))

	with open(os.path.join(os.getcwd(), 'testTrain_chr13.fa'), 'w') as fs:

		for idx, val in enumerate(data):

			pairIdx = (idx + random.randint(MIN_PAIR, MAX_PAIR)) % (len(seq) - MAX_PAIR - SEQLENGTH)
			pair = data[pairIdx]

			fs.write('\t'.join([val, pair, idx, pairIdx]) + '\n')