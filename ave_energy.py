"""
calculate the average ground energy from train_stats.csv by averaging the last 100 steps.
"""

import subprocess
import numpy as np
import sys

process = subprocess.run("tail -100 train_stats.csv", shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
last_100_lines = process.stdout.split('\n')[0:int(sys.argv[1])]
get_100_energy = [float(x.split(',')[1]) for x in last_100_lines]

ave_energy = np.mean(get_100_energy)
print(f"average ground state energy: {ave_energy}")
