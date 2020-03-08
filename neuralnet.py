import numpy as np

class NeuralNet:
    def __init__(self, structure):
        self.structure = structure
        self.num_layers = len(structure)
        self.params = self.init_layers()
        self.cache = {}
        self.grads = {}

    def init_layers(self):
        params = {}

        for index, layer in enumerate(self.structure):
            layer_index = index + 1
            input_size = layer['input_dim']
            output_size = layer['output_dim']

            params["W" + str(layer_index)] = np.random.randn(output_size, input_size) * 0.01
            params["b" + str(layer_index)] = np.zeros((output_size, 1))

        return params

    def train(self, X, Y, num_iterations = 10000, print_cost = False):
        for i in range(num_iterations):
            A = self.forward_propagation(X)
            cost = compute_cost(A, Y, self.params)

            self.backward_propagation(A, Y)
            self.update_parameters()

            if print_cost and i % 1000 == 0:
                print(f"Cost after iteration {i}: {cost}")

    def predict(self, X):
        yHat = self.forward_propagation(X)
        return yHat

    def forward_propagation(self, X):
        A = X
        self.cache["A0"] = X

        for index, layer in enumerate(self.structure):
            layer_index = index + 1
            activation = layer['activation']

            A = self._forward_layer(A, layer_index, activation)
        
        return A

    def _forward_layer(self, A, layer, activation = "relu"):
        W = self.params["W" + str(layer)]
        b = self.params["b" + str(layer)]

        Z = np.dot(W, A) + b

        if activation == "relu":
            A = relu(Z)
        elif activation == "sigmoid":
            A = sigmoid(Z)

        self.cache["Z" + str(layer)] = Z
        self.cache["A" + str(layer)] = A

        return A

    def backward_propagation(self, yHat, Y):
        m = Y.shape[1]

        last_layer = self.num_layers

        dZ = (-2 * (Y - yHat)) * sigmoid_derivative(self.cache["Z" + str(last_layer)])
        
        for index_prev, layer in reversed(list(enumerate(self.structure))):
            index_curr = index_prev + 1
            activation = layer["activation"]

            A_prev = self.cache["A" + str(index_prev)]

            dW = (1 / m) * np.dot(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            self.grads["dW" + str(index_curr)] = dW
            self.grads["db" + str(index_curr)] = db

            if index_prev == 0: break

            W = self.params["W" + str(index_curr)]
            Z = self.cache["Z" + str(index_prev)]
            if activation == "relu":
                dZ = np.dot(W.T, dZ) * relu_derivative(Z)
            elif activation == "sigmoid":
                dZ = np.dot(W.T, dZ) * sigmoid_derivative(Z)

    def update_parameters(self, learning_rate = 0.1):
        for index, layer in enumerate(self.structure):
            layer_index = index + 1

            W = self.params["W" + str(layer_index)]
            b = self.params["b" + str(layer_index)]

            dW = self.grads["dW" + str(layer_index)]
            db = self.grads["db" + str(layer_index)]

            self.params["W" + str(layer_index)] = W - learning_rate * dW
            self.params["b" + str(layer_index)] = b - learning_rate * db


def compute_cost(yHat, Y, parameters):
    m = Y.shape[1]
    error = (1 / 2) * (np.linalg.norm(Y - yHat, axis=0) ** 2)
    cost = (1 / m) * np.sum(error)

    return cost

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return np.exp(-z) / (np.power(1 + np.exp(-z), 2))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return 1. * (z > 0)