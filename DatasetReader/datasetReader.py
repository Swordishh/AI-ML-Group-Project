import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("../NN/polyRegression/1950-2024_all_tornadoes.csv")
pd.set_option('display.max_rows', None)
fT = df[df["mag"]< 5]
fd = fT[fT["mag"]>=0]
filtered = fd[fd["yr"]>= 1995]


def plotCount(x, y):
    R = filtered.groupby(x)[y].count().plot(kind="bar", figsize=(12,4))
    plt.xlabel(f"{x}")
    plt.ylabel(f"{y}")
    plt.title(f"Count:{x}/{y}")
    plt.show()
    return R

def plotAvg(x, y):
    R = filtered.groupby(x)[y].mean().plot(kind="bar", figsize=(12,4))
    plt.xlabel(f"{x}")
    plt.ylabel(f"{y}")
    plt.title(f"Mean:{x}/{y}")
    plt.show()
    return R

def plotStd(x, y):
    R = filtered.groupby(x)[y].std().plot(kind="bar", figsize=(12,4))
    plt.xlabel(f"{x}")
    plt.ylabel(f"{y}")
    plt.title(f"Standard Deviation:{x}/{y}")
    plt.show()
    return R

def plots(x, y):
    rList = [plotCount(x, y), plotAvg(x, y), plotStd(x, y)]

plots("yr", "mag")
# plots("st", "mag")
# plots("closs", "mag")

