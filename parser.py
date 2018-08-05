import os
import sys

if __name__ == '__main__':

	seq = dict()

	with open(sys.argv[1], 'r') as fs:

		raw = fs.readlines()

	for line in raw:

		if 'chr' in line:

			name = line.replace('\n', '')[1:]
			seq[name] = list()

		else:

			seq[name].append(line.replace('\n', ''))

	for key, val in seq.items():

		seq[key] = ''.join(val)

	for key, val in seq.items():

		with open(os.path.join(os.getcwd(), key + '.fa'), 'w') as fs:

			fs.write(val)