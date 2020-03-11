import numpy as np

def sigmoid():
    def forward(Z):
        A = 1 / (1 + np.exp(-Z))
        return A, Z

    def backward(dA, cache):
        sig = 1 / (1 + np.exp(-cache))
        dZ = dA * sig * (1 - sig)
        return dZ

    return forward, backward

def softmax():
    def forward(Z):
        A = np.exp(Z - np.max(Z, axis=0))
        A = A / np.sum(A, axis=0)
        return A, Z

    def backward(dA, cache):
        Z = cache
        A = np.exp(Z - np.max(Z, axis=0))
        A = A / np.sum(A, axis=0)

        S = np.tile(A, S.shape[0])
        return np.diag(A) - (S * S.T)

    return forward, backward

def tanh():
    def forward(Z):
        A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        return A, Z
    
    def backward(dA, cache):
        Z = cache
        tanh = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        dZ = 1 - np.power(tanh, 2)
        return dZ

    return forward, backward

def relu():
    def forward(Z):
        A = np.maximum(0, Z)
        return A, Z

    def backward(dA, cache):
        dZ = np.array(dA, copy=True)
        dZ[cache <= 0] = 0
        return dZ

    return forward, backward