import numpy as np
import matplotlib.pyplot as plt

def get_fake_data(iter,show=False):
    X = np.random.rand(iter)*20
    noise = np.random.randn(iter)
    y = 0.5 * X + noise
    if show:
        plt.scatter(X,y)
        plt.show()
    return X,y

def count_y_prediction(X, w, b):
    y_pred = w*X + [b]
    # print(y_pred)
    return y_pred

def compete_error_for_given_points(y, y_pred):
    error = (y - y_pred) ** 2
    error = error.sum() / 2
    # print(error)
    return error

def compete_gradient_and_update(X, w, b, lr):
    w_gradient = 0
    b_gradient = 0
    N = len(X)
    for i in range(N):
        w_gradient += 2 * (w * X[i] + b - y[i]) * X[i]
        b_gradient += 2 *(w * X[i] + b - y[i])
    w -= lr * w_gradient / N
    b -= lr * b_gradient / N
    return w,b
def update2(y, w, b, lr):
    w = 0.5
    b_d = 0
    for i in y:
        print(b,i)
        b_d +=(b-i)
    b =b-(lr*b_d)
    print("db:",b_d)
    return w,b
def update(w, b, lr):
    w = 0.5
    b +=(lr*200)
    return w,b
losss= []
def linaerRegression(X, y, w, b, i, lr = 0.001):
    # print(w,b)
    y_pred = count_y_prediction(X, w, b)
    error = compete_error_for_given_points(y, y_pred)
    print("loss:", error)
    losss.append(error)

    # w, b = compete_gradient_and_update(X, w, b, lr)
    w, b = update2(y, w, b, lr)
    print(w,b)
    y_pred = count_y_prediction(X, w, b)
    # draw(X, y, y_pred)
    return w,b

def draw(X, y, y_pred,final=True):
    # plt.ion()
    plt.clf()
    plt.scatter(X, y, c="blue")
    plt.plot(X, y_pred, c="blue")
    if final:
        plt.pause(0.2) 
        # plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    iter = 50
    X,y = get_fake_data(iter,True)
    w = np.random.randn(1)
    b = -10
    plt.ion() 
    for i in range(30):
        w,b = linaerRegression(X, y, w, b, i,0.001)
    plt.ioff()
    plt.scatter(list(range(30)),losss)
    plt.show()
    # y_pred = count_y_prediction(X, w, b)
    # draw(X,y,y_pred,0)
