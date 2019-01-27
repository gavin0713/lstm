import pandas as pd
import numpy as np

df = pd.read_csv("datatest")

value = df["test"].values

def getdata(data, scale, windows=True):
    ret = None
    if(windows):
        ret = np.zeros(len(data) - scale)
        for i in range(len(ret)):
            ret[i] = np.sum(data[i:i + scale])
    else:
        ret = np.zeros(int(len(data) / scale))
        for i in range(len(ret)):
            ret[i] = np.sum(data[i * scale:(i + 1) * scale])

    return ret

if __name__ == '__main__':
    data = getdata(value, 180)

    fout = open("outdata", "w")

    for d in data:
        fout.write(str(d))
        fout.write("\n")

    fout.close()
