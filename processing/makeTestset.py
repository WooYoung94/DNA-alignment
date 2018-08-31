# makeTestset.py
# 염색체 단위로 나눠진 fa 파일에서 N 을 포함하지 않은 시퀀스에 대해서
# 시작 index 와 32bp 만큼의 시퀀스를 pair 로 하는 데이터셋 생성
# 이때, 일정 확률로 시퀀스를 변형하여 저장함 (snp 를 의도)

import os
import sys
import random

SEQLENGTH = 32
MIN_PAIR = 300
MAX_PAIR = 5000
SNP_PROB = 0.001

def makeVariant(seq):

	# SNP
	idx = random.randint(0, 31)
	seq = seq.upper()
	baseList = ['A', 'G', 'T', 'C']
	baseList.remove(seq[idx])
	seq = seq[:idx] + random.choice(baseList) + seq[idx + 1:]

	return seq

def getSeqence(seq, idx, length):

	return seq[idx:idx + length]

if __name__ == '__main__':

	data = list()
	dataVariant = list()

	with open(sys.argv[1], 'r') as fs:

		seq = fs.read()
		print(len(seq))

	print(len(seq[:-SEQLENGTH]) + 1)

	for idx in range(len(seq[:-SEQLENGTH]) + 1):

		cut = getSeqence(seq, idx, SEQLENGTH)

		if 'N' not in cut:

			data.append(cut)

	print(len(data))

	for idx, val in enumerate(data):

		prob = random.random()

		if prob < SNP_PROB:

			variant = makeVariant(val)

			if variant in data:

				dataVariant.append(val)

			else:

				dataVariant.append(variant)

		else:

			dataVariant.append(val)

	with open(os.path.join(os.getcwd(), 'variant_' + os.path.basename(sys.argv[1])), 'w') as fs:

		for idx, val in enumerate(dataVariant):

			#gap = random.randint(MIN_PAIR, MAX_PAIR)
			#pairIdx = (idx + gap) % (len(seq) - MAX_PAIR - SEQLENGTH)
			#pair = data[pairIdx]

			#fs.write('\t'.join([val, pair, str(gap), str(idx), str(pairIdx)]) + '\n')

			fs.write('\t'.join([val, str(idx)]) + '\n')
