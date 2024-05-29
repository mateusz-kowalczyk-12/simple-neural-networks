import math
import sys

import numpy as np


def cross_entropy(class_idx, output_layer):
    # class_idx - index of the class with respect to which the softmax is calculates

    sm = softmax(class_idx, output_layer)
    return -math.log(sm) if sm > 0 else sys.float_info.max


def softmax(class_idx, output_layer):
    # class_idx - index of the class with respect to which the softmax is calculates

    denominator = 0
    for a_i in output_layer.a:
        denominator += math.exp(a_i)

    return math.exp(output_layer.a[class_idx]) / denominator

