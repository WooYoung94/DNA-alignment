# makeTestset.py
# 염색체 단위로 나눠진 fa 파일에서 N 을 포함하지 않은 시퀀스에 대해서
# 시작 index 와 32bp 만큼의 시퀀스를 pair 로 하는 데이터셋 생성
# 이때, 일정 확률로 시퀀스를 변형하여 저장함 (snp 를 의도)

import os
import sys
import random

SEQLENGTH = 32
SNP_PROB = 0.01

if __name__ == '__main__':

	with open(sys.argv[1], 'r') as fs:

		full = fs.read()
		print(len(full))

	full = list(full)
	variantIdx = [random.randint(0, len(full)) for _ in range(int(SNP_PROB * len(full)))]

	for idx in range(len(full)):

		if full[idx] != 'N' and idx in variantIdx:

			full[idx] = 'W'

	with open(os.path.join(os.getcwd(), 'variant_' + os.path.basename(sys.argv[1])), 'w') as fs:

		fs.write(''.join(full))