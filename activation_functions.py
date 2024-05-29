import math
import sys


def sigmoid(z: float):
    return 1 / (1 + math.exp(-z))


def d_sigmoid(z: float):  # derivative
    return math.exp(z) / ((1 + math.exp(z))**2)


def identity(z: float):
    return z


def d_identity(z: float):
    return 1
