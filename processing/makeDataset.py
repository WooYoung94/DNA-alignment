import os
import sys
import random

SEQLENGTH = 32
MIN_PAIR = 300
MAX_PAIR = 5000

def getSeqence(seq, idx, length):

	return seq[idx:idx + length]

if __name__ == '__main__':

	data = list()

	with open(sys.argv[1], 'r') as fs:

		seq = fs.read()
		print(len(seq))

	print(len(seq[:-SEQLENGTH]) + 1)

	for idx in range(len(seq[:-SEQLENGTH]) + 1):

		cut = getSeqence(seq, idx, SEQLENGTH)

		if 'N' not in cut:

			data.append(cut)

	print(len(data))

	with open(os.path.join(os.getcwd(), 'test_' + os.path.basename(sys.argv[1]), 'w')) as fs:

		for idx, val in enumerate(data):

			#gap = random.randint(MIN_PAIR, MAX_PAIR)
			#pairIdx = (idx + gap) % (len(seq) - MAX_PAIR - SEQLENGTH)
			#pair = data[pairIdx]

			#fs.write('\t'.join([val, pair, str(gap), str(idx), str(pairIdx)]) + '\n')

			fs.write('\t'.join(val, str(idx) + '\n'))
