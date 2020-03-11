import numpy as np


class NeuralNet:
    def __init__(self, layers, activations, cost):
        self.layers = layers
        self.activations = [()] + activations
        self.num_layers = len(layers)
        self.params = self._init_layers()
        self.cost, self.cost_derivative = cost

        assert(len(layers) == len(activations) + 1)

    def _init_layers(self):
        params = {}

        for l in range(1, self.num_layers):
            params['W' + str(l)] = np.random.randn(self.layers[l],
                                                   self.layers[l - 1]) * 0.01
            params['b' + str(l)] = np.zeros((self.layers[l], 1))

        return params

    def _forward_linear(self, A, W, b):
        Z = np.dot(W, A) + b
        cache = (A, W, b)

        return Z, cache

    def _forward_activation(self, A_prev, W, b, activation):
        Z, linear_cache = self._forward_linear(A_prev, W, b)
        A, activation_cache = activation(Z)

        cache = (linear_cache, activation_cache)
        return A, cache

    def forward_propagation(self, X):
        caches = []
        A = X

        for l in range(1, self.num_layers):
            A_prev = A

            W = self.params['W' + str(l)]
            b = self.params['b' + str(l)]
            activation, _ = self.activations[l]

            A, cache = self._forward_activation(A_prev, W, b, activation)
            caches.append(cache)

        return A, caches

    def _backward_linear(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def _backward_activation(self, dA, cache, activation):
        linear_cache, activation_cache = cache

        dZ = activation(dA, activation_cache)
        dA_prev, dW, db = self._backward_linear(dZ, linear_cache)

        return dA_prev, dW, db

    def backward_propagation(self, AL, Y, caches):
        grads = {}
        m = AL.shape[1]

        dAL = self.cost_derivative(AL, Y)
        grads['dA' + str(self.num_layers - 1)] = dAL

        for l in reversed(range(self.num_layers - 1)):
            layer = l + 1
            cache = caches[l]
            _, activation = self.activations[layer]
            dA, dW, db = self._backward_activation(grads['dA' + str(layer)], cache, activation)

            grads['dA' + str(l)] = dA
            grads['dW' + str(layer)] = dW
            grads['db' + str(layer)] = db

        return grads

    def _update_params(self, grads, learning_rate):
        for l in range(self.num_layers - 1):
            self.params['W' + str(l + 1)] = self.params['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
            self.params['b' + str(l + 1)] = self.params['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]

    def train(self, X, Y, num_iterations=10000, learning_rate=0.1, print_cost=False):
        costs = []

        for i in range(num_iterations):
            A, caches = self.forward_propagation(X)
            cost = self.cost(A, Y)
            costs.append(cost)

            grads = self.backward_propagation(A, Y, caches)
            self._update_params(grads, learning_rate)

            if print_cost and i % 1000 == 0:
                print(f"Cost after iteration {i} = {cost}")
        return costs

    def predict(self, X):
        AL, _ = self.forward_propagation(X)
        return AL

    def reset(self):
        self.params = self._init_layers()


def meansquared():
    def cost(AL, Y):
        m = Y.shape[1]

        error = (1 / 2) * (np.linalg.norm(Y - AL, axis=0) ** 2)
        cost = (1 / m) * np.sum(error)
        return cost

    def derivative(AL, Y):
        dAL = (-2 * (Y - AL))
        return dAL

    return cost, derivative


def crossentropy():
    def cost(AL, Y):
        m = Y.shape[1]

        cost = (-1 / m) * np.sum((Y * np.log(AL + 1e-15) + ((1 - Y) * np.log(1 - AL + 1e-15))))
        cost = np.squeeze(cost)
        return cost

    def derivative(AL, Y):
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        return dAL

    return cost, derivative
