import sys
import os
import subprocess
import numpy as np


RUN_SCRIPT = 'run_video.py'
RANDOM_SEED = 42
RUN_TIME = 20  # sec
ABR_ALGO = ['fastMPC', 'robustMPC', 'BOLA', 'RL']
REPEAT_TIME = 10


def main():

	np.random.seed(RANDOM_SEED)

	with open('./chrome_retry_log', 'wb') as log:
		log.write('chrome retry log\n')
		log.flush()

		for rt in xrange(REPEAT_TIME):
			np.random.shuffle(ABR_ALGO)
			for abr_algo in ABR_ALGO:

				while True:

					script = 'python ' + RUN_SCRIPT + ' ' + \
							  abr_algo + ' ' + str(RUN_TIME) + ' ' + str(rt)
					
					print "running" + script
					
					proc = subprocess.Popen(script,
							  stdout=subprocess.PIPE, 
							  stderr=subprocess.PIPE, 
							  shell=True)

					(out, err) = proc.communicate()

					

					if out == 'done\n':
						print "proc finished by done"
						break
					else:
						print "proc finished by other"
						log.write(abr_algo + '_' + str(rt) + '\n')
						log.write('starting output \n')
						log.write(out + '\n')
						log.write('ending output \n')
						log.write(err + '\n')
						log.flush()
					break


if __name__ == '__main__':
	main()
