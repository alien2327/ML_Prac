import csv
import numpy as np

if __name__ == "__main__":
    fname = "_result.csv"
    res_fname = "stat.csv"

    max_gen = []
    times = []

    for i in range(100):
        fname_1 = str(i)+fname
        with open(fname_1) as f:
            reader = csv.reader(f)
            gen = 0
            t = []
            for row in reader:
                gen +=1
                t.append(row[-1])
            max_gen.append(gen-1)
            times.append(float(t[-1]))

    max_gen = np.array(max_gen)
    mean_gen = np.mean(max_gen)
    std_gen = np.std(max_gen)
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    with open(res_fname, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([mean_gen, std_gen, mean_time, std_time])