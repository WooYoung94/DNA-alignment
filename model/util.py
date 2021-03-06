import os
import sys

if __name__ == '__main__':

	log = list()

	if sys.argv[1] == 'loss':

		with open(sys.argv[2], 'r') as fs:

			text = fs.read().split('=')[0].split('\n')[:-1]

		for t in text:

			log.append(float(t.split(':')[-1].strip()))

	elif sys.argv[1] == 'pred':

		with open(sys.argv[2], 'r') as fs:

			text = fs.read().split('=')[1].split('\n')[:-1]

		for t in text:

			l1 = int(t.split(',')[0].split(':')[-1].strip().split('/')[0][2:-1])
			l2 = int(t.split(',')[0].split(':')[-1].strip().split('/')[1][1:-2])
			log.append((l1, l2))

		log.sort()

	elif sys.argv[1] == 'delta':

		with open(sys.argv[2], 'r') as fs:

			text = fs.read().split('=')[1].split('\n')[:-1]

		for t in text:

			l1 = int(t.split(',')[0].split(':')[-1].strip().split('/')[0][2:-1])
			l2 = int(t.split(',')[0].split(':')[-1].strip().split('/')[1][1:-2])
			log.append((l1, l2))

		log.sort()
		log = [l[0]-l[1] for l in log]

	print(log)