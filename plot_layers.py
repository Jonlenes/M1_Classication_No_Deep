from plot_function import PlotFunction

import numpy as np

x = []
y = []

x.append(np.loadtxt("layers.txt"))
y.append(np.loadtxt("accs.txt"))

x.append(np.loadtxt("layers2.txt"))
y.append(np.loadtxt("accs2.txt"))

x.append(np.loadtxt("layers3.txt"))
y.append(np.loadtxt("accs3.txt"))

plt = PlotFunction()

for i, j in zip(x, y):
    plt.add_function(i, j)

plt.show()
