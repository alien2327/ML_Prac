import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import mean

if __name__ == "__main__":
    f1 = "stat_indiv_size.csv"
    f2 = "stat_mut_rate.csv"
    f3 = "stat_tournament_size.csv"

    mean_gen = []
    std_gen = []
    mean_time = []
    std_time = []

    with open(f3) as f:
        r = csv.reader(f)
        for row in r:
            mean_gen.append(float(row[0]))
            std_gen.append(float(row[1]))
            mean_time.append(float(row[2]))
            std_time.append(float(row[3]))

    fig = plt.figure()
    x = np.linspace(100, 100+50*len(mean_gen), len(mean_gen))

    ax = fig.add_subplot(111)
    
    ax2 = ax.twinx()
    ax.set_title("Comparison with Tournament size and Max Generation/Time Duration", fontsize=30)
    ax.errorbar(x, mean_gen, yerr=std_gen, color='b', label='Mean Max Generation')
    ax.set_xlabel("Tournament size", size=30)
    ax.set_ylabel("Max Generation", size=30)

    ax2.errorbar(x, mean_time, yerr=std_time, color='g', label='Mean Time Duration')
    ax2.set_ylabel("Time Duration[ms]", size=30)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, fontsize=30)

    ax.tick_params(axis = 'both', labelsize=30)
    ax2.tick_params(axis = 'both', labelsize=30)

    ax.grid()

    plt.show()