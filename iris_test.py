from neuralnet import NeuralNet, compute_cost
from visualize import plot_cost, learning_rate_test

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

structure = [
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 3, "activation": "sigmoid"},
]

def prepare_data(X, y):
    y = np.reshape(y, (y.shape[0], 1))
    class_num = len(np.unique(y))

    y_classes = np.zeros((y.shape[0], class_num))
    for i in range(class_num):
        y_classes[:,i] = np.where(y == i, 1, 0).reshape((y.shape[0]))

    X_train, X_test, y_train, y_test = train_test_split(X, y_classes)

    X_train = X_train.T
    X_test = X_test.T
    y_train = y_train.T
    y_test = y_test.T

    return (X_train, X_test, y_train, y_test)

def compute_accuracy(net, X, y):
    predictions = net.predict(X)
    predictions = np.where(predictions > 0.5, 1, 0)

    accuracy = (predictions == y).all(axis=0).mean()
    return f"{(accuracy * 100):.2f}"

def main():
    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    net = NeuralNet(structure)
    #learning_rate_test(net, X_train, y_train, [0.5, 0.1, 0.05, 0.01])

    learning_rate = 0.1
    net = NeuralNet(structure)
    costs = net.train(X_train, y_train, learning_rate=learning_rate, print_cost=True)

    accuracy = compute_accuracy(net, X_test, y_test)
    print(f"\nModel Accuracy: {accuracy}%\n")

    plot_cost(costs, learning_rate)

if __name__ == '__main__':
    main()