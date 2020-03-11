from neuralnet import NeuralNet, crossentropy, meansquared
from activations import sigmoid, relu, tanh
from visualize import plot_decision_boundary, learning_rate_test

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def prepare_data():
    X, Y = load_iris(return_X_y=True)
    Y = Y.reshape((Y.shape[0], 1))
    class_num = len(np.unique(Y))

    y_classes = np.zeros((Y.shape[0], class_num))
    for i in range(class_num):
        y_classes[:, i] = np.where(Y == i, 1,  0).reshape((Y.shape[0]))

    X_train, X_test, y_train, y_test = train_test_split(X, y_classes)

    return X_train.T, X_test.T, y_train.T, y_test.T

def compute_accuracy(net, X, y):
    predictions = net.predict(X)
    predictions = np.where(predictions > 0.5, 1, 0)

    accuracy = (predictions == y).all(axis=0).mean()
    print(f"Accuracy = {(accuracy * 100):.2f}%")

def main():
    X_train, X_test, y_train, y_test = prepare_data()

    layers = [4, 6, 3]
    activations = [relu(), sigmoid()]

    net = NeuralNet(layers, activations, meansquared())
    #learning_rate_test(net, X_train, y_train, [0.2, 0.1, 0.05, 0.01])
    #return

    costs = net.train(X_train, y_train, learning_rate=0.2, print_cost=True)

    compute_accuracy(net, X_test, y_test)


if __name__ == '__main__':
    main()