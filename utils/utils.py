import collections
import matplotlib.pyplot as plt
import pandas as pd


def printFrequency(arr):
    freq = collections.Counter(arr)

    for key, value in freq.items():
        print(key, " -> ", value)

    return freq

def plotFrequencyBar(freq):
    plt.bar(range(len(freq)), list(freq.values()), align='center')
    plt.xticks(range(len(freq)), list(freq.keys()))
    plt.show()

def cmp(a, b):
    return (a > b) - (a < b)

def csvToDictionary(pathToCsv):
    df = pd.read_csv(pathToCsv, index_col=0)
    breedToFrquency = df['breed'].value_counts().to_dict()
    s = [(k, breedToFrquency[k]) for k in sorted(breedToFrquency, key=breedToFrquency.get, reverse=False)]
    for k, v in s:
        print(k,v)


#csvToDictionary('C:\work\ml\work\cypherlabsimagesearch\data\dogbreed\labels.csv')