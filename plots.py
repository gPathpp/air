import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    for file, baseline in [('train_loss.csv', 'cos_train_loss.csv'), ('test_loss.csv', 'cos_test_loss.csv')]:
        df = pd.read_csv(file)
        df = df.iloc[:, 1:]
        df.plot(xlabel='epoch', ylabel='loss')

        best_loss_baseline = np.min(pd.read_csv(baseline).iloc[:, 1:].to_numpy())

        minimum = np.min(df.values)
        index = np.where(df.min() == minimum)[0]
        plt.hlines(y=best_loss_baseline, xmin=-5, xmax=105, label="Best baseline")
        plt.scatter(df[::-1].idxmin()[index][0], minimum, c='r')
        plt.legend()
        plt.savefig('plots/' + file[:-4] + '.png')
        plt.show()

    for file, baseline in [('train_f1.csv', 'cos_train_f1.csv'), ('test_f1.csv', 'cos_test_f1.csv')]:
        df = pd.read_csv(file)
        df = df.iloc[:, 1:]
        df.plot(xlabel='epoch', ylabel='f1 score')

        best_f1_baseline = np.max(pd.read_csv(baseline).iloc[:, 1:].to_numpy())

        maximum = np.max(df.values)
        index = np.where(df.max() == maximum)[0]
        plt.hlines(y=best_f1_baseline, xmin=-5, xmax=105, label="Best baseline")
        plt.scatter(df[::-1].idxmax()[index][0], maximum, c='r')
        plt.legend()
        plt.savefig('plots/' + file[:-4] + '.png')
        plt.show()
