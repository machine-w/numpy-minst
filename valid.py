
import numpy as np
from nn_fun import predict
import matplotlib.pyplot as plt
from pathlib import Path
PATH = Path('./model')

train_loss_list=[]
valid_loss_list=[]
train_accu_list=[]
valid_accu_list=[]
def sqr_loss(img,lab,parameters):
    y_pred=predict(img,parameters)
    y=np.identity(10)[lab]
    diff=y-y_pred
    return np.dot(diff,diff)
def valid_loss(parameters,data):
    loss_accu=0
    for img_i in range(data.valid_num):
        loss_accu+=sqr_loss(data.valid_img[img_i],data.valid_lab[img_i],parameters)
    return loss_accu/(data.valid_num/10000)
def valid_accuracy(parameters,data):
    correct=[predict(data.valid_img[img_i],parameters).argmax()==data.valid_lab[img_i] for img_i in range(data.valid_num)]
    return correct.count(True)/len(correct)
def train_loss(parameters,data):
    loss_accu=0
    for img_i in range(data.train_num):
        loss_accu+=sqr_loss(data.train_img[img_i],data.train_lab[img_i],parameters)
    return loss_accu/(data.train_num/10000)
def train_accuracy(parameters,data):
    correct=[predict(data.train_img[img_i],parameters).argmax()==data.train_lab[img_i] for img_i in range(data.train_num)]
    return correct.count(True)/len(correct)


def valid_log(parameters,data):
    train_loss_list.append(train_loss(parameters,data))
    train_accu_list.append(train_accuracy(parameters,data))
    valid_loss_list.append(valid_loss(parameters,data))
    valid_accu_list.append(valid_accuracy(parameters,data))

def show_loss_log():
    plt.plot(valid_loss_list, c='g', label='validation loss')
    plt.plot(train_loss_list, c='b', label='train loss')
    plt.legend()
    plt.savefig(PATH/'loss_view')
    plt.clf()

def show_acc_log():
    plt.plot(valid_accu_list, c='g', label='validation acc')
    plt.plot(train_accu_list, c='b', label='train acc')
    plt.legend()
    plt.savefig(PATH/'accuracy_view')
    plt.clf()