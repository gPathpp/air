import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    for file in ['train_loss.csv', 'test_loss.csv']:
        df = pd.read_csv(file)
        df = df.iloc[:, 1:]
        df.plot(xlabel='epoch', ylabel='loss')

        minimum = np.min(df.values)
        index = np.where(df.min() == minimum)[0]
        plt.scatter(df[::-1].idxmin()[index][0], minimum, c='r')
        plt.savefig('plots/' + file[:-4] + '.png')
        plt.show()

    for file in ['train_f1.csv', 'test_f1.csv']:
        df = pd.read_csv(file)
        df = df.iloc[:, 1:]
        df.plot(xlabel='epoch', ylabel='f1 score')

        maximum = np.max(df.values)
        index = np.where(df.max() == maximum)[0]
        plt.scatter(df[::-1].idxmax()[index][0], maximum, c='r')
        plt.savefig('plots/' + file[:-4] + '.png')
        plt.show()
