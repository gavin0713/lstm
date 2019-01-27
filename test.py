import pandas as pd
import numpy as np

def nash(a, b):
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    c = np.zeros(len(a))
    d = np.zeros(len(a))
    for i in range(len(d) - 1):
        c[i] = (a[i] - b[i]) ** 2
        d[i] = (a[i] - a_mean) ** 2
    c_sum = np.sum(c)
    d_sum = np.sum(d)
    return 1 - c_sum / d_sum

if __name__ == '__main__':

    df = pd.read_table('hydrodata')
    datalist = df['流量'].values

    print(nash(datalist[1:], datalist[:-1]))

