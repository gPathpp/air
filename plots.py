import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    for file in ['train_loss.csv', 'train_f1.csv', 'test_loss.csv', 'test_f1.csv']:
        df = pd.read_csv(file)
        df = df.iloc[:, 1:]
        p = df.plot(xlabel='epoch', ylabel=file[:-4])
        plt.savefig('plots/' + file[:-4] + '.png')
        plt.show()
