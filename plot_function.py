import matplotlib.pyplot as plt
import numpy as np


class PlotFunction:
    count = 0

    def __init__(self):
        self.style = ['r', 'g--', 'b--', 'g--', 'b', 'r--']

    def add_function(self, x_plot, y_plot):
        plt.plot(x_plot, y_plot, self.style[self.count], label="Modelo " + str(self.count + 1))
        self.count += 1

    def show(self):
        plt.legend(loc=4)
        plt.xlabel('Layers')
        plt.ylabel('Precisão')
        # plt.yscale('log')
        # plt.yticks([10 ** 24 * (10 ** 26) ** i for i in range(11)])
        plt.grid(True)
        plt.savefig("/home/jonlenes/Desktop/figura_2.png")
        plt.show()
