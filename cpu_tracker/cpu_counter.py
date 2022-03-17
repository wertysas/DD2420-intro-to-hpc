"""
Author: Johan Ericsson
Date: 2022-03-17
"""


import argparse
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from cpu_tracker import CPUTracker



def plot_cpu_usage(tracker: CPUTracker):
    """
    Plots the cpu usage of each core
    """
    assert len(tracker.percents) > 0, "No data recorded by CPUTracker"
    n_physical_threads = tracker.physical_threads
    n_physical_cores = tracker.physical_cores


    #
    percents = np.array(tracker.percents)
    # We plot with a maximum of 8 threads/row
    n_rows = ceil(n_physical_cores/4)
    fig, axes = plt.subplots(n_rows, 4, sharex='all', sharey='all', figsize=(16, 12))
    axes = axes.flatten()
    for i in range(0, n_physical_threads, 2):
        ax_ind = i//2
        axes[ax_ind].plot(percents[:, i], ":", label='Thread: '+str(i), linewidth=2.0)
        axes[ax_ind].plot(percents[:, i+1], ":", label='Thread: ' + str(i+1), linewidth=2.0)
        axes[ax_ind].legend()
    fig.suptitle('CPU consumption per phsyical thread')
    fig.supylabel('CPU consumption per thread (% of max)')
    fig.tight_layout()


def arg_parsing():
    parser = argparse.ArgumentParser("CPU Threads measurement tool")
    parser.add_argument("--interval", dest="update_interval", type=int,
                        help="update interval, by default 0.5 (recommended)")
    parser.add_argument("--time", dest="measurement_time", type=int,
                        help="running time in seconds, default set to 10")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Input parsing
    parser = argparse.ArgumentParser(prog="CPUCounter CPU measurement tool")
    parser.add_argument("--interval", dest="update_interval", type=float,
                        help="update interval (sec), by default 0.5 (recommended)", nargs="?",
                        const=1, default=0.5)
    parser.add_argument("--time", dest="measurement_time", type=int,
                        help="running time (sec), default set to 10", nargs="?",
                        const=1, default=10)
    args = parser.parse_args()

    # Measurement Code
    update_interval = args.update_interval
    n_updates = int(args.measurement_time/update_interval)
    tracker = CPUTracker(update_interval)
    for i in range(n_updates):
        tracker.update()
    plot_cpu_usage(tracker)
    plt.show()

