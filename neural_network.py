import matplotlib.pyplot as plt
import numpy as np

from activation_functions import sigmoid, d_sigmoid, identity, d_identity
from data_plotter import DataPlotter
from output_functions import cross_entropy, softmax


class NeuralNetwork:
    def __init__(self, type_idx):
        if type_idx == 0:
            self.layers_n = 3
            self.neurons_n = np.array([2, 3, 1])  # an array of neurons numbers in each layer
            self.layers = np.array([
                Layer(self.neurons_n[l_idx], self.neurons_n[l_idx - 1] if l_idx > 0 else 0, sigmoid, d_sigmoid)
                for l_idx in range(self.layers_n)
            ])
        if type_idx == 1:
            self.layers_n = 4
            self.neurons_n = np.array([1, 2, 2, 1])  # an array of neurons numbers in each layer
            self.layers = np.array([
                Layer(self.neurons_n[l_idx], self.neurons_n[l_idx - 1] if l_idx > 0 else 0, sigmoid, d_sigmoid)
                for l_idx in range(self.layers_n)
            ])
        if type_idx == 2:
            self.layers_n = 3
            self.neurons_n = np.array([2, 3, 2])  # an array of neurons numbers in each layer
            self.layers = np.array([
                Layer(self.neurons_n[l_idx], self.neurons_n[l_idx - 1] if l_idx > 0 else 0, sigmoid, d_sigmoid)
                for l_idx in range(self.layers_n)
            ])
            self.layers[self.layers_n - 1].activation_function = identity
            self.layers[self.layers_n - 1].d_activation_function = d_identity

    def train(self, X_train, Y_train, X_test, eta, epochs, type_idx):
        for i in range(epochs):
            self.backpropagate(X_train, Y_train, eta, type_idx, i)

            if i == 0 or (i + 1) % 1000 == 0:
                print(f"Epoch: {i + 1}")
                predictions = np.array([
                    self.predict(x, 0)
                    for x in X_test
                ])
                DataPlotter.plot_single_stripe_data(
                    X_test, predictions,
                    np.array([np.array([0, 1]), np.array([0, 1])]),
                    'Predictions', 'x', 'y', i + 1)

    def predict(self, x, type_idx):
        self.propagate(x)
        if type_idx == 0:
            return 0 if self.layers[2].a[0] < 0.5 else 1
        if type_idx == 1:
            return self.layers[3].a[0]
        if type_idx == 2:
            return np.argmax(np.array([
                softmax(i, self.layers[2])
                for i in range(2)
            ]))

    def propagate(self, x):
        self.layers[0].a = x
        for l_idx in range(1, self.layers_n):
            self.layers[l_idx].z = self.layers[l_idx].w @ self.layers[l_idx - 1].a + self.layers[l_idx].b
            self.layers[l_idx].a = np.array([
                self.layers[l_idx].activation_function(self.layers[l_idx].z[i])
                for i in range(self.layers[l_idx].z.shape[0])
            ])

    def backpropagate(self, X_train, Y_train, eta, type_idx, epoch_idx):
        cost = 0

        if type_idx == 0:
            # cost functions derivatives
            dC_db_0_l2 = 0.
            dC_dw_0_0_l2 = 0.
            dC_dw_0_1_l2 = 0.
            dC_dw_0_2_l2 = 0.
            dC_db_0_l1 = 0.
            dC_dw_0_0_l1 = 0.
            dC_dw_0_1_l1 = 0.
            dC_db_1_l1 = 0.
            dC_dw_1_0_l1 = 0.
            dC_dw_1_1_l1 = 0.
            dC_db_2_l1 = 0.
            dC_dw_2_0_l1 = 0.
            dC_dw_2_1_l1 = 0.

            for k in range(X_train.shape[0]):
                self.propagate(X_train[k])

                cost += (self.layers[2].a[0] - Y_train[k])**2

                dC_db_0_l2 += \
                    2 * (self.layers[2].a[0] - Y_train[k]) * d_sigmoid(self.layers[2].z[0])
                dC_dw_0_0_l2 += \
                    2 * (self.layers[2].a[0] - Y_train[k]) * d_sigmoid(self.layers[2].z[0]) * self.layers[1].a[0]
                dC_dw_0_1_l2 += \
                    2 * (self.layers[2].a[0] - Y_train[k]) * d_sigmoid(self.layers[2].z[0]) * self.layers[1].a[1]
                dC_dw_0_2_l2 += \
                    2 * (self.layers[2].a[0] - Y_train[k]) * d_sigmoid(self.layers[2].z[0]) * self.layers[1].a[2]
                dC_db_0_l1 += \
                    2 * (self.layers[2].a[0] - Y_train[k]) * d_sigmoid(self.layers[2].z[0]) * self.layers[2].w[0][0] \
                    * d_sigmoid(self.layers[1].z[0])
                dC_dw_0_0_l1 += \
                    2 * (self.layers[2].a[0] - Y_train[k]) * d_sigmoid(self.layers[2].z[0]) * self.layers[2].w[0][0] \
                    * d_sigmoid(self.layers[1].z[0]) * self.layers[0].a[0]
                dC_dw_0_1_l1 += \
                    2 * (self.layers[2].a[0] - Y_train[k]) * d_sigmoid(self.layers[2].z[0]) * self.layers[2].w[0][0] \
                    * d_sigmoid(self.layers[1].z[0]) * self.layers[0].a[1]
                dC_db_1_l1 += \
                    2 * (self.layers[2].a[0] - Y_train[k]) * d_sigmoid(self.layers[2].z[0]) * self.layers[2].w[0][1] \
                    * d_sigmoid(self.layers[1].z[1])
                dC_dw_1_0_l1 += \
                    2 * (self.layers[2].a[0] - Y_train[k]) * d_sigmoid(self.layers[2].z[0]) * self.layers[2].w[0][1] \
                    * d_sigmoid(self.layers[1].z[1]) * self.layers[0].a[0]
                dC_dw_1_1_l1 += \
                    2 * (self.layers[2].a[0] - Y_train[k]) * d_sigmoid(self.layers[2].z[0]) * self.layers[2].w[0][1] \
                    * d_sigmoid(self.layers[1].z[1]) * self.layers[0].a[1]
                dC_db_2_l1 += \
                    2 * (self.layers[2].a[0] - Y_train[k]) * d_sigmoid(self.layers[2].z[0]) * self.layers[2].w[0][2] \
                    * d_sigmoid(self.layers[1].z[2])
                dC_dw_2_0_l1 += \
                    2 * (self.layers[2].a[0] - Y_train[k]) * d_sigmoid(self.layers[2].z[0]) * self.layers[2].w[0][2] \
                    * d_sigmoid(self.layers[1].z[2]) * self.layers[0].a[0]
                dC_dw_2_1_l1 += \
                    2 * (self.layers[2].a[0] - Y_train[k]) * d_sigmoid(self.layers[2].z[0]) * self.layers[2].w[0][2] \
                    * d_sigmoid(self.layers[1].z[2]) * self.layers[0].a[1]

            if epoch_idx == 0 or (epoch_idx + 1) % 1000 == 0:
                print(f"Cost: {cost}")

            self.layers[2].b[0] -= dC_db_0_l2 * eta
            self.layers[2].w[0][0] -= dC_dw_0_0_l2 * eta
            self.layers[2].w[0][1] -= dC_dw_0_1_l2 * eta
            self.layers[2].w[0][2] -= dC_dw_0_2_l2 * eta
            self.layers[1].b[0] -= dC_db_0_l1 * eta
            self.layers[1].w[0][0] -= dC_dw_0_0_l1 * eta
            self.layers[1].w[0][1] -= dC_dw_0_1_l1 * eta
            self.layers[1].b[1] -= dC_db_1_l1 * eta
            self.layers[1].w[1][0] -= dC_dw_1_0_l1 * eta
            self.layers[1].w[1][1] -= dC_dw_1_1_l1 * eta
            self.layers[1].b[2] -= dC_db_2_l1 * eta
            self.layers[1].w[2][0] -= dC_dw_2_0_l1 * eta
            self.layers[1].w[2][1] -= dC_dw_2_1_l1 * eta
        if type_idx == 1:
            # cost functions derivatives
            dC_db_0_l3 = 0.
            dC_dw_0_0_l3 = 0.
            dC_dw_0_1_l3 = 0.
            dC_db_0_l2 = 0.
            dC_dw_0_0_l2 = 0.
            dC_dw_0_1_l2 = 0.
            dC_db_1_l2 = 0.
            dC_dw_1_0_l2 = 0.
            dC_dw_1_1_l2 = 0.
            dC_db_0_l1 = 0.
            dC_dw_0_0_l1 = 0.
            dC_db_1_l1 = 0.
            dC_dw_1_0_l1 = 0.

            for k in range(X_train.shape[0]):
                self.propagate(X_train[k])

                cost += (self.layers[3].a[0] - Y_train[k]) ** 2

                dC_db_0_l3 += \
                    1 * d_sigmoid(self.layers[3].z[0]) * 2 * (self.layers[3].a[0] - Y_train[k])
                dC_dw_0_0_l3 += \
                    self.layers[2].a[0] * d_sigmoid(self.layers[3].z[0]) * 2 * (self.layers[3].a[0] - Y_train[k])
                dC_dw_0_1_l3 += \
                    self.layers[2].a[1] * d_sigmoid(self.layers[3].z[0]) * 2 * (self.layers[3].a[0] - Y_train[k])
                dC_db_0_l2 +=\
                    1 * d_sigmoid(self.layers[2].z[0]) * self.layers[3].w[0][0]\
                    * d_sigmoid(self.layers[3].z[0]) * 2 * (self.layers[3].a[0] - Y_train[k])
                dC_dw_0_0_l2 += \
                    self.layers[1].a[0] * d_sigmoid(self.layers[2].z[0]) * self.layers[3].w[0][0] \
                    * d_sigmoid(self.layers[3].z[0]) * 2 * (self.layers[3].a[0] - Y_train[k])
                dC_dw_0_1_l2 += \
                    self.layers[1].a[1] * d_sigmoid(self.layers[2].z[0]) * self.layers[3].w[0][0] \
                    * d_sigmoid(self.layers[3].z[0]) * 2 * (self.layers[3].a[0] - Y_train[k])
                dC_db_1_l2 += \
                    1 * d_sigmoid(self.layers[2].z[1]) * self.layers[3].w[0][1] \
                    * d_sigmoid(self.layers[3].z[0]) * 2 * (self.layers[3].a[0] - Y_train[k])
                dC_dw_1_0_l2 += \
                    self.layers[1].a[0] * d_sigmoid(self.layers[2].z[1]) * self.layers[3].w[0][1] \
                    * d_sigmoid(self.layers[3].z[0]) * 2 * (self.layers[3].a[0] - Y_train[k])
                dC_dw_1_1_l2 += \
                    self.layers[1].a[1] * d_sigmoid(self.layers[2].z[1]) * self.layers[3].w[0][1] \
                    * d_sigmoid(self.layers[3].z[0]) * 2 * (self.layers[3].a[0] - Y_train[k])
                dC_db_0_l1 += \
                    1 * d_sigmoid(self.layers[1].z[0]) * \
                    (self.layers[2].w[0][0] * d_sigmoid(self.layers[2].z[0]) * self.layers[3].w[0][0] +
                     self.layers[2].w[1][0] * d_sigmoid(self.layers[2].z[1]) * self.layers[3].w[0][1]) * \
                    d_sigmoid(self.layers[3].z[0]) * 2 * (self.layers[3].a[0] - Y_train[k])
                dC_dw_0_0_l1 += \
                    self.layers[0].a[0] * d_sigmoid(self.layers[1].z[0]) * \
                    (self.layers[2].w[0][0] * d_sigmoid(self.layers[2].z[0]) * self.layers[3].w[0][0] +
                     self.layers[2].w[1][0] * d_sigmoid(self.layers[2].z[1]) * self.layers[3].w[0][1]) * \
                    d_sigmoid(self.layers[3].z[0]) * 2 * (self.layers[3].a[0] - Y_train[k])
                dC_db_1_l1 += \
                    1 * d_sigmoid(self.layers[1].z[1]) * \
                    (self.layers[2].w[0][1] * d_sigmoid(self.layers[2].z[0]) * self.layers[3].w[0][0] +
                     self.layers[2].w[1][1] * d_sigmoid(self.layers[2].z[1]) * self.layers[3].w[0][1]) * \
                    d_sigmoid(self.layers[3].z[0]) * 2 * (self.layers[3].a[0] - Y_train[k])
                dC_dw_1_0_l1 += \
                    self.layers[0].a[0] * d_sigmoid(self.layers[1].z[1]) * \
                    (self.layers[2].w[0][1] * d_sigmoid(self.layers[2].z[0]) * self.layers[3].w[0][0] +
                     self.layers[2].w[1][1] * d_sigmoid(self.layers[2].z[1]) * self.layers[3].w[0][1]) * \
                    d_sigmoid(self.layers[3].z[0]) * 2 * (self.layers[3].a[0] - Y_train[k])

            if epoch_idx % 500 == 0:
                print(f"Cost: {cost}")

            self.layers[3].b[0] -= dC_db_0_l3 * eta
            self.layers[3].w[0][0] -= dC_dw_0_0_l3 * eta
            self.layers[3].w[0][1] -= dC_dw_0_1_l3 * eta
            self.layers[2].b[0] -= dC_db_0_l2 * eta
            self.layers[2].w[0][0] -= dC_dw_0_0_l2 * eta
            self.layers[2].w[0][1] -= dC_dw_0_1_l2 * eta
            self.layers[2].b[1] -= dC_db_1_l2 * eta
            self.layers[2].w[1][0] -= dC_dw_1_0_l2 * eta
            self.layers[2].w[1][1] -= dC_dw_1_1_l2 * eta
            self.layers[1].b[0] -= dC_db_0_l1 * eta
            self.layers[1].w[0][0] -= dC_dw_0_0_l1 * eta
            self.layers[1].b[1] -= dC_db_1_l1 * eta
            self.layers[1].w[1][0] -= dC_dw_1_0_l1 * eta
        if type_idx == 2:
            # cost functions derivatives
            dC_db_0_l2 = 0.
            dC_dw_0_0_l2 = 0.
            dC_dw_0_1_l2 = 0.
            dC_dw_0_2_l2 = 0.
            dC_db_1_l2 = 0.
            dC_dw_1_0_l2 = 0.
            dC_dw_1_1_l2 = 0.
            dC_dw_1_2_l2 = 0.
            dC_db_0_l1 = 0.
            dC_dw_0_0_l1 = 0.
            dC_dw_0_1_l1 = 0.
            dC_db_1_l1 = 0.
            dC_dw_1_0_l1 = 0.
            dC_dw_1_1_l1 = 0.
            dC_db_2_l1 = 0.
            dC_dw_2_0_l1 = 0.
            dC_dw_2_1_l1 = 0.

            for k in range(X_train.shape[0]):
                self.propagate(X_train[k])

                cost += cross_entropy(Y_train[k], self.layers[2])

                dC_db_0_l2 += 1 * (
                    softmax(0, self.layers[2]) - 1 if Y_train[k] == 0
                    else softmax(0, self.layers[2])
                )
                dC_dw_0_0_l2 += self.layers[1].a[0] * (
                    softmax(0, self.layers[2]) - 1 if Y_train[k] == 0
                    else softmax(0, self.layers[2])
                )
                dC_dw_0_1_l2 += self.layers[1].a[1] * (
                    softmax(0, self.layers[2]) - 1 if Y_train[k] == 0
                    else softmax(0, self.layers[2])
                )
                dC_dw_0_2_l2 += self.layers[1].a[2] * (
                    softmax(0, self.layers[2]) - 1 if Y_train[k] == 0
                    else softmax(0, self.layers[2])
                )
                dC_db_1_l2 += 1 * (
                    softmax(1, self.layers[2]) - 1 if Y_train[k] == 1
                    else softmax(1, self.layers[2])
                )
                dC_dw_1_0_l2 += self.layers[1].a[0] * (
                    softmax(1, self.layers[2]) - 1 if Y_train[k] == 1
                    else softmax(1, self.layers[2])
                )
                dC_dw_1_1_l2 += self.layers[1].a[1] * (
                    softmax(1, self.layers[2]) - 1 if Y_train[k] == 1
                    else softmax(1, self.layers[2])
                )
                dC_dw_1_2_l2 += self.layers[1].a[2] * (
                    softmax(1, self.layers[2]) - 1 if Y_train[k] == 1
                    else softmax(1, self.layers[2])
                )
                dC_db_0_l1 += 1 * self.layers[1].d_activation_function(self.layers[1].z[0]) * (
                    self.layers[2].w[0][0] * (
                        softmax(0, self.layers[2]) - 1 if Y_train[k] == 0
                        else softmax(0, self.layers[2])
                    ) +
                    self.layers[2].w[1][0] * (
                        softmax(1, self.layers[2]) - 1 if Y_train[k] == 1
                        else softmax(1, self.layers[2])
                    )
                )
                dC_dw_0_0_l1 += self.layers[0].a[0] * self.layers[1].d_activation_function(self.layers[1].z[0]) * (
                    self.layers[2].w[0][0] * (
                        softmax(0, self.layers[2]) - 1 if Y_train[k] == 0
                        else softmax(0, self.layers[2])
                    ) +
                    self.layers[2].w[1][0] * (
                        softmax(1, self.layers[2]) - 1 if Y_train[k] == 1
                        else softmax(1, self.layers[2])
                    )
                )
                dC_dw_0_1_l1 += self.layers[0].a[1] * self.layers[1].d_activation_function(self.layers[1].z[0]) * (
                    self.layers[2].w[0][0] * (
                        softmax(0, self.layers[2]) - 1 if Y_train[k] == 0
                        else softmax(0, self.layers[2])
                    ) +
                    self.layers[2].w[1][0] * (
                        softmax(1, self.layers[2]) - 1 if Y_train[k] == 1
                        else softmax(1, self.layers[2])
                    )
                )
                dC_db_1_l1 += 1 * self.layers[1].d_activation_function(self.layers[1].z[1]) * (
                    self.layers[2].w[0][1] * (
                        softmax(0, self.layers[2]) - 1 if Y_train[k] == 0
                        else softmax(0, self.layers[2])
                    ) +
                    self.layers[2].w[1][1] * (
                        softmax(1, self.layers[2]) - 1 if Y_train[k] == 1
                        else softmax(1, self.layers[2])
                    )
                )
                dC_dw_1_0_l1 += self.layers[0].a[0] * self.layers[1].d_activation_function(self.layers[1].z[0]) * (
                    self.layers[2].w[0][1] * (
                        softmax(0, self.layers[2]) - 1 if Y_train[k] == 0
                        else softmax(0, self.layers[2])
                    ) +
                    self.layers[2].w[1][1] * (
                        softmax(1, self.layers[2]) - 1 if Y_train[k] == 1
                        else softmax(1, self.layers[2])
                    )
                )
                dC_dw_1_1_l1 += self.layers[0].a[1] * self.layers[1].d_activation_function(self.layers[1].z[0]) * (
                    self.layers[2].w[0][1] * (
                        softmax(0, self.layers[2]) - 1 if Y_train[k] == 0
                        else softmax(0, self.layers[2])
                    ) +
                    self.layers[2].w[1][1] * (
                        softmax(1, self.layers[2]) - 1 if Y_train[k] == 1
                        else softmax(1, self.layers[2])
                    )
                )
                dC_db_2_l1 += 1 * self.layers[1].d_activation_function(self.layers[1].z[2]) * (
                    self.layers[2].w[0][2] * (
                        softmax(0, self.layers[2]) - 1 if Y_train[k] == 0
                        else softmax(0, self.layers[2])
                    ) +
                    self.layers[2].w[1][2] * (
                        softmax(1, self.layers[2]) - 1 if Y_train[k] == 1
                        else softmax(1, self.layers[2])
                    )
                )
                dC_dw_2_0_l1 += self.layers[0].a[0] * self.layers[1].d_activation_function(self.layers[1].z[0]) * (
                    self.layers[2].w[0][2] * (
                        softmax(0, self.layers[2]) - 1 if Y_train[k] == 0
                        else softmax(0, self.layers[2])
                    ) +
                    self.layers[2].w[1][2] * (
                        softmax(1, self.layers[2]) - 1 if Y_train[k] == 1
                        else softmax(1, self.layers[2])
                    )
                )
                dC_dw_2_1_l1 += self.layers[0].a[1] * self.layers[1].d_activation_function(self.layers[1].z[0]) * (
                    self.layers[2].w[0][2] * (
                        softmax(0, self.layers[2]) - 1 if Y_train[k] == 0
                        else softmax(0, self.layers[2])
                    ) +
                    self.layers[2].w[1][2] * (
                        softmax(1, self.layers[2]) - 1 if Y_train[k] == 1
                        else softmax(1, self.layers[2])
                    )
                )

            if epoch_idx % 100 == 0:
                print(f"Cost: {cost}")

            self.layers[2].b[0] -= dC_db_0_l2 * eta
            self.layers[2].w[0][0] -= dC_dw_0_0_l2 * eta
            self.layers[2].w[0][1] -= dC_dw_0_1_l2 * eta
            self.layers[2].w[0][2] -= dC_dw_0_2_l2 * eta
            self.layers[2].b[1] -= dC_db_1_l2 * eta
            self.layers[2].w[1][0] -= dC_dw_1_0_l2 * eta
            self.layers[2].w[1][1] -= dC_dw_1_1_l2 * eta
            self.layers[2].w[1][2] -= dC_dw_1_2_l2 * eta
            self.layers[1].b[0] -= dC_db_0_l1 * eta
            self.layers[1].w[0][0] -= dC_dw_0_0_l1 * eta
            self.layers[1].w[0][1] -= dC_dw_0_1_l1 * eta
            self.layers[1].b[1] -= dC_db_1_l1 * eta
            self.layers[1].w[1][0] -= dC_dw_1_0_l1 * eta
            self.layers[1].w[1][1] -= dC_dw_1_1_l1 * eta
            self.layers[1].b[2] -= dC_db_2_l1 * eta
            self.layers[1].w[2][0] -= dC_dw_2_0_l1 * eta
            self.layers[1].w[2][1] -= dC_dw_2_1_l1 * eta


class Layer:
    def __init__(self, neurons_n, prev_neurons_n, activation_function, d_activation_function):
        self.neurons_n = neurons_n
        self.prev_neurons_n = prev_neurons_n  # number of neurons in the previous layer

        self.a = np.zeros(neurons_n)
        self.z = np.zeros(neurons_n) if prev_neurons_n > 0 else None
        self.w = np.random.rand(neurons_n, prev_neurons_n) if prev_neurons_n > 0 else None
        self.b = np.zeros(neurons_n) if prev_neurons_n > 0 else None
        self.activation_function = activation_function if prev_neurons_n > 0 else None
        self.d_activation_function = d_activation_function if prev_neurons_n > 0 else None
