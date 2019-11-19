import numpy as np
import matplotlib.pyplot as plt


def initialize_parameters(lenv):
    w = np.random.rand(1, len)
    b = 0
    return w, b


def forward_prop(X, w, b):
    z = np.dot(w, X) + b  # b_vector = [b b b ...]
    return z


def cost_function(z, y):
    m = y.shape[1]
    J = (1 / (2 * m)) * np.sum(np.square(z - y))
    return J


def back_prop(X, y, z):
    m = y.shape[1]
    dz = (1 / m) * (z - y)
    dw = np.dot(dz, X.T)  # dw->1xn
    db = np.sum(dz)
    return dw, db


def gradient_descent_update(w, b, dw, db, learning_rate):
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b

