import numpy as np
import pickle

from data_generator import DataGenerator
from data_plotter import DataPlotter
from neural_network import NeuralNetwork


# neural_network0 - layers: (2, 3, 1)
# neural_network1 - layers: (1, 2, 2, 1)
# neural_network2 - layers: (2, 3, 2, SoftMax)


def main():
    neural_network0 = NeuralNetwork(0)

    x_lims = np.array([np.array([0, 1]), np.array([0, 1])])
    X_train, Y_train, X_test, Y_test = DataGenerator.generate_single_stripe_data(
        train_size=1000, test_size=1000,
        x_lims=x_lims
    )
    with open("models/nn0.pkl", "wb") as file:
        pickle.dump(neural_network0, file, pickle.HIGHEST_PROTOCOL)

    # X_train, Y_train, X_test, Y_test = DataGenerator.generate_single_square_data(
    #     train_size=1000, test_size=1000,
    #     x_lims=x_lims
    # )
    DataPlotter.plot_single_stripe_data(X_train, Y_train, x_lims, 'Train data', 'x_0', 'x_1', 'train')
    DataPlotter.plot_single_stripe_data(X_test, Y_test, x_lims, 'Test data', 'x_0', 'x_1', 'test')

    neural_network0.train(X_train, Y_train, X_test, 0.01, 10000, 0)
    predictions = np.array([
        neural_network0.predict(x, 0)
        for x in X_test
    ])

    DataPlotter.plot_single_stripe_data(X_test, predictions, x_lims, 'Predictions', 'x_0', 'x_1', None)

    # neural_network1 = NeuralNetwork(1)
    # X_train, Y_train, X_test = DataGenerator.generate_tower_points_data()
    #
    # neural_network1.train(X_train, Y_train, X_test, 0.1, 10000, 1)
    # predictions = np.array([
    #     neural_network1.predict(x, 1)
    #     for x in X_test
    # ])
    # DataPlotter.plot_tower_points_data(X_test, predictions, 'Predictions', 'x', 'y', 10000)

    # neural_network2 = NeuralNetwork(2)
    #
    # x_lims = np.array([np.array([0, 0.8]), np.array([0, 0.8])])
    # X_train, Y_train, X_test, Y_test = DataGenerator.generate_single_stripe_data(
    #     train_size=1000, test_size=1000,
    #     x_lims=x_lims
    # )
    # DataPlotter.plot_single_stripe_data(X_train, Y_train, x_lims, 'Train data', 'x_0', 'x_1', None)
    # DataPlotter.plot_single_stripe_data(X_test, Y_test, x_lims, 'Test data', 'x_0', 'x_1', None)

    # neural_network2.train(X_train, Y_train, X_test, 0.001, 100000, 2)
    # predictions = np.array([
    #     neural_network2.predict(x, 2)
    #     for x in X_test
    # ])
    # DataPlotter.plot_single_stripe_data(X_test, predictions, x_lims, 'Predictions', 'x_0', 'x_1', None)


if __name__ == '__main__':
    main()
