import sys
from IPython.core.ultratb import ColorTB
sys.excepthook = ColorTB()

from rich import print
import random
import pandas as pd
import numpy as np
from pyutils import eda
import matplotlib.pyplot as plt

class RandomWalker:
    def __init__(self,n):
        self.position = 0            
        self.n = n

    def __len__(self):
        return len(self.n)
    
    def walk(self):
        self.position = 0        
        for i in range(self.n):
            yield self.position
            self.position += 2*random.randint(0, 1) - 1


walker = RandomWalker(10)

plt.plot([[position for position in RandomWalker(100).walk()] for i in range(4)])
plt.show()

ax1 = df.A.plot(color='blue', grid=True, label='Count')
ax2 = df.B.plot(color='red', grid=True, secondary_y=True, label='Sum')


from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D

dims = 1
step_n = 10000
step_set = [-1, 0, 1]
origin = np.zeros((1,dims))
# Simulate steps in 1D
step_shape = (step_n,dims)
steps = np.random.choice(a=step_set, size=step_shape)
path = np.concatenate([origin, steps]).cumsum(0)
start = path[:1]
stop = path[-1:]
# Plot the path
fig = plt.figure(figsize=(8,4),dpi=200)
ax = fig.add_subplot(111)
ax.scatter(np.arange(step_n+1), path,alpha=0.25,s=0.05);
ax.plot(path,alpha=0.5,lw=0.5);
plt.show()

pd.Series(np.ravel(path)).plot()
plt.show()

slow_time = 15.7
fast_time = 2.2


1-(fast_time/slow_time)

from pyutils import load
path_ending_with_filename = 'D:/Downloads/tabular-playground-series-apr-2021'
load.read_data(path_ending_with_filename)
load.read_data(path_ending_with_filename,dataframes='sample_submission,train,test',return_df=True, method='dt')

dataframes = 'sample_submission,train'
dataframes = [x.strip(" ") for x in dataframes.split(",")]
csvs_in_directory = [x for x in os.listdir(path_ending_with_filename) if x.endswith('.csv')]
files = list(set(csvs_in_directory) & set([x + '.csv' for x in dataframes]))








