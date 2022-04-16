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
    1
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
colors = cycle(‘bgrcmykbgrcmykbgrcmykbgrcmyk’)

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
ax.scatter(np.arange(step_n+1), path, c=’blue’,alpha=0.25,s=0.05);
ax.plot(path,c=’blue’,alpha=0.5,lw=0.5,ls=’ — ‘,);
ax.plot(0, start, c=’red’, marker=’+’)
ax.plot(step_n, stop, c=’black’, marker=’o’)

