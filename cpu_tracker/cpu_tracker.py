"""
Author: Johan Ericsson
Date: 2022-03-12
"""

import psutil
import numpy as np
import matplotlib.pyplot as plt


class CPUTracker:
    def __init__(self, update_interval=0.5):
        self.interval = update_interval
        self.percents = []
        self.physical_cores = psutil.cpu_count(logical=False)
        self.physical_threads = psutil.cpu_count(logical=True)

    def update(self):
        self.percents.append(psutil.cpu_percent(interval=self.interval, percpu=True))

    def plot_cpu_usage(self):
        t = [self.interval*i for i in range(len(self.percents))]
        y = np.array(self.percents)
        plt.plot(t, y)

    def live_plot(self):
        """
        if time
        """
        pass
