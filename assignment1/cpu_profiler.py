"""
Author: Johan Ericsson
Date: 2022-03-12
"""

import psutil
import numpy as np
import matplotlib.pyplot as plt

class CPUTracker:
    def __init__(self):
        self.interval = 0.5
        self.percents = []

    def update(self):
        self.percents.append(psutil.cpu_percent(interval=1, percpu=True))

    def plot_cpu_usage(self):
        t = [self.interval*i for i in range(len(self.percents))]
        y = np.array(self.percents)
        plt.plot(t, y)


if __name__ == '__main__':
    tracker = CPUTracker()
    for i in range(20):
        tracker.update()
    tracker.plot_cpu_usage()
    plt.show()
