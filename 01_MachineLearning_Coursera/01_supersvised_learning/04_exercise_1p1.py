import time
import numpy as np
import matplotlib.pyplot as plt


def get_cost(x, y, w, b):
    m = len(x)
    if len(y) != m:
        raise ValueError("Review dataset")

    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2
    cost = 1 / 2 / m * cost

    return cost


def get_gradient(x, y, w, b):
    m = len(x)

    dJ_dw = 0
    dJ_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dJ_dw += (f_wb - y[i]) * x[i]
        dJ_db += f_wb - y[i]
    dJ_dw = dJ_dw / m
    dJ_db = dJ_db / m

    return dJ_dw, dJ_db


def gradient_descent(x, y, w_0, b_0, alpha, num_iterations):
    w_history = []
    b_history = []
    J_history = []

    w, b = w_0, b_0
    i = 0
    while True:
        cost = get_cost(x, y, w, b)
        dJ_dw, dJ_db = get_gradient(x, y, w, b)

        w -= alpha * dJ_dw
        b -= alpha * dJ_db

        w_history.append(w)
        b_history.append(b)
        J_history.append(cost)

        i += 1
        if i == num_iterations:
            break

    return w_history, b_history, J_history


if __name__ == "__main__":
    x_train = []
    y_train = []
    with open('data.txt', 'r') as data_file:
        for line in data_file:
            data = line.strip().split(',')
            x_train.append(float(data[0]))
            y_train.append(float(data[2]))

    ALPHA = 1.0e-8
    ITER = int(1e3)
    w_0, b_0 = 0, 0
    start = time.time()
    w_hist, b_hist, J_hist = gradient_descent(x_train, y_train,
                                              w_0, b_0, ALPHA, ITER)
    end = time.time()
    elapsed_time = end - start
    print(f"Elapsed time: {round(1000 * elapsed_time, 2)} ms")

    x_train_array = np.array(x_train)
    y_train_array = np.array(y_train)
    y_prediction = w_hist[-1] * x_train_array + b_hist[-1]
    err_mean = np.sum((y_train_array - np.mean(y_train_array) ** 2))
    err_pred = np.sum((y_train_array - y_prediction) ** 2)
    r_squared = 1 - err_pred / err_mean
    print(f"R-squared: {r_squared}")

    plt.scatter(x_train, y_train, color='blue')
    plt.plot(x_train_array, y_prediction, color='red')
    plt.xlabel('x_data')
    plt.ylabel('y_data')
    plt.savefig('data.png')
