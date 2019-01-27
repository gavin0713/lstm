import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# a实测
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



    df = pd.read_table('out2')

    df.columns = ['a', 'b']

    a = df.a.values;
    b = df.b.values;


    print(nash(a,b))

    trainlen = 365 * 2
    print(nash(a[:trainlen], b[:trainlen]))
    print(nash(a[trainlen:], b[trainlen:]))


    # nashlist = []
    # for i in range(1,5000):
    #     nashlist += [nash(a[i:], b[:-i])]
    #
    # plt.plot(nashlist)
    #
    # plt.show()
    # print(nash(a[1:], b[:-1]))
    #
    # print(nash(b[1:], a[:-1]))
    #
    # print(nash(a[2:], b[:-2]))
    #
    # print(nash(b[3:], a[:-3]))
    #
    # print(nash(b[4:], a[:-4]))
    #
    # print(nash(a[5:], b[:-5]))
    #
    # print(nash(b[6:], a[:-6]))
    #
    # print(nash(b[7:], a[:-7]))
    #
    # print(nash(a[8:], b[:-8]))
    #
    # print(nash(b[9:], a[:-9]))
    #
    # print(nash(b[10:], a[:-10]))
    #
    # print(nash(a[11:], b[:-11]))
    #
    # print(nash(b[12:], a[:-12]))
    #
    # print(nash(b[20:], a[:-20]))
    #
    # print(nash(b[30:], a[:-30]))
    # print(nash(b[40:], a[:-40]))
    #
