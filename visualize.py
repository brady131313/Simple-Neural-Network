import matplotlib.pyplot as plt

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
        costs.append(cost)
    plot_costs(costs, rates)