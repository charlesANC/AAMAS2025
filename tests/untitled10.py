'''
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

data = np.random.rand(10, 10) * 20

# create discrete colormap
cmap = colors.ListedColormap(['red', 'green', 'yellow', 'blue'])
bounds = [0,5,10,15,20]
norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots()
ax.imshow(data, cmap=cmap, norm=norm)

# draw gridlines
ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
ax.set_xticks(np.arange(-.5, 10, 1));
ax.set_yticks(np.arange(-.5, 10, 1));

plt.show()
'''

import random

agent_names = ['AgentA', 'AgentB']
agent_pos = [0, 4, 8, 12]


pos = { agent_names[i]: agent_pos[i] for i in range(len(agent_names)) }

print(pos)

print(agent_names.index('AgentB'))

