import matplotlib.pyplot as plt
import numpy as np


class DataPlotter:
    @staticmethod
    def plot_single_stripe_data(X, Y, x_lims, title, xlabel, ylabel, filename_idx):
        for i in range(X.shape[0]):
            plt.scatter(X[i][0], X[i][1], s=20, c='orange' if Y[i] == 0 else 'blue')

        plt.plot([0, 0.4], [0.4, 0], c='black')
        plt.plot([0, 0.7], [0.7, 0], c='black')

        plt.xlim(x_lims[0])
        plt.ylim(x_lims[1])

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # plt.show()
        plt.savefig(f"plots/nn0_{filename_idx}")
        plt.clf()

    @staticmethod
    def plot_single_square_data(X, Y, x_lims, title, xlabel, ylabel):
        for i in range(X.shape[0]):
            plt.scatter(X[i][0], X[i][1], s=20, c='orange' if Y[i] == 0 else 'blue')

        plt.plot([1, 2.5], [2, 0.5], c='black')
        plt.plot([2.5, 4], [0.5, 2], c='black')
        plt.plot([4, 2.5], [2, 3.5], c='black')
        plt.plot([2.5, 1], [3.5, 2], c='black')

        plt.xlim(x_lims[0])
        plt.ylim(x_lims[1])

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.show()

    @staticmethod
    def plot_tower_points_data(X_test, predictions, title, xlabel, ylabel, epoch_idx):
        plt.plot(X_test, predictions, c='blue')
        plt.scatter([0., 0.25, 0.5, 1.], [0., 0.25, 1., 0.], s=20, c='black')

        plt.xlim([-0.5, 1.5])
        plt.ylim([-0.5, 1.5])

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.savefig(f"plots/nn1_{epoch_idx}")
        plt.clf()
