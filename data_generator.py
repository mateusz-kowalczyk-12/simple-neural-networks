import numpy as np


class DataGenerator:
    # x_lims[i][0] - min of i-th coordinate
    # x_lims[i][1] - max of i-th coordinate

    @staticmethod
    def generate_single_stripe_data(train_size, test_size, x_lims):
        X = DataGenerator.generate_2D_X(train_size, test_size, x_lims)
        Y = np.array([
            0 if (-X[i][0] + 0.4 < X[i][1] < -X[i][0] + 0.7) else 1
            for i in range(X.shape[0])
        ])

        return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

    @staticmethod
    def generate_single_square_data(train_size, test_size, x_lims):
        X = DataGenerator.generate_2D_X(train_size, test_size, x_lims)
        Y = np.array([
            0 if (-X[i][0] + 3 < X[i][1] < -X[i][0] + 6 and X[i][0] -2 < X[i][1] < X[i][0] + 1) else 1
            for i in range(X.shape[0])
        ])

        return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

    @staticmethod
    def generate_tower_points_data():
        X_train = np.array([[0.], [0.25], [0.5], [1.]] * 10)
        Y_train = np.array([0., 0.25, 1., 0.] * 10)
        X_test = np.array([np.array([x])for x in np.linspace(0, 1, 100)])

        return X_train, Y_train, X_test

    @staticmethod
    def generate_2D_X(train_size, test_size, x_lims):
        X = np.array([
            np.array([
                np.random.rand() * (x_lims[0][1] - x_lims[0][0]) + x_lims[0][0],  # 0-th coordinate
                np.random.rand() * (x_lims[1][1] - x_lims[1][0]) + x_lims[1][0]  # 1-st coordinate
            ])
            for _ in range(train_size + test_size)
        ])

        return X
