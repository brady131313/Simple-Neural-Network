from neuralnet import NeuralNet, compute_cost

import numpy as np
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

def main():
    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    print(X_train.shape, y_train.shape)

    net = NeuralNet(structure)
    net.train(X_train, y_train, print_cost=True)

    predictions = net.predict(X_test)
    predictions = np.where(predictions > 0.5, 1, 0)

    np.set_printoptions(linewidth=130)
    
    accuracy = (predictions == y_test).all(axis=0).mean()
    print(accuracy)
    print(y_test[:,5])
    print(predictions[:,5])

if __name__ == '__main__':
    main()