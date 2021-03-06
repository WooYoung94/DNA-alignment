# makeDataset.py
# 염색체 단위로 나눠진 fa 파일에서 N 을 포함하지 않은 시퀀스에 대해서
# 시작 index 와 32bp 만큼의 시퀀스를 pair 로 하는 데이터셋 생성 

import os
import sys
import random

SEQLENGTH = 32

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

	with open(os.path.join(os.getcwd(), 'trainData_' + os.path.basename(sys.argv[1])), 'w') as fs:

		for idx, val in enumerate(data):

			fs.write('\t'.join([val, str(idx)]) + '\n')