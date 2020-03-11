import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(net, X, y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = net.predict(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)

    y = y.reshape((y.shape[1]))

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.T, cmap=plt.cm.Spectral)
    plt.show()

def plot_cost(costs, learning_rate):
    x = range(len(costs))

    fig, ax = plt.subplots()
    ax.plot(x, costs, label=str(learning_rate))

    ax.set(xlabel="Iteration", ylabel="Cost", title="Cost After Each Iteration of Gradient Descent")
    ax.grid()
    ax.legend(title="Learning Rates")

    plt.show()

def plot_costs(costs, learning_rates):
    fig, ax = plt.subplots()

    for cost, rate in zip(costs, learning_rates):
        x = range(len(cost))
        ax.plot(x, cost, label=str(rate))


    ax.set(xlabel="Iteration", ylabel="Cost", title="Cost After Each Iteration of Gradient Descent")
    ax.grid()
    ax.legend(title="Learning Rates")

    plt.show()

def learning_rate_test(net, X, y, rates):
    costs = []
    for rate in rates:
        cost = net.train(X, y, learning_rate=rate)
        net.reset()
        costs.append(cost)
    plot_costs(costs, rates)