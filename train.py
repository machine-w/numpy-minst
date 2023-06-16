import numpy as np
import copy
import struct
import pickle

from oper_data import TrainData
from nn_fun import tanh,d_softmax,d_tanh,softmax
from init_param import init_parameters,init_parameters_simple
from valid import valid_log,show_loss_log,show_acc_log
from pathlib import Path
PATH = Path('./model')
#导入数据
data = TrainData()
data.open_file()
#反向传播
def grad_parameters(img,lab,parameters):
    l0_in=img+parameters[0]['b']
    l0_out=tanh(l0_in)
    l1_in=np.dot(l0_out,parameters[1]['w'])+parameters[1]['b']
    l1_out=softmax(l1_in)
    
    diff=np.identity(10)[lab]-l1_out
    act1=np.dot(d_softmax(l1_in),diff)
    
    grad_b1=-2*act1
    grad_w1=-2*np.outer(l0_out,act1)
    grad_b0=-2*d_tanh(l0_in)*np.dot(parameters[1]['w'],act1)
    
    return {'w1':grad_w1,'b1':grad_b1,'b0':grad_b0}
#梯度下降
def combine_parameters(parameters,grad,learn_rate):
    parameter_tmp=copy.deepcopy(parameters)
    parameter_tmp[0]['b']-=learn_rate*grad['b0']
    parameter_tmp[1]['b']-=learn_rate*grad['b1']
    parameter_tmp[1]['w']-=learn_rate*grad['w1']
    return parameter_tmp
#################################################################训练
learn_rate=10**-0.6
# learn_rate=1

parameters=init_parameters_simple()
epoch_num=30
current_epoch=1
batch_size=100
def train_batch(current_batch,parameters):
    grad_accu=grad_parameters(data.train_img[current_batch*batch_size+0],data.train_lab[current_batch*batch_size+0],parameters)
    for img_i in range(1,batch_size):
        grad_tmp=grad_parameters(data.train_img[current_batch*batch_size+img_i],data.train_lab[current_batch*batch_size+img_i],parameters)
        for key in grad_accu.keys():
            grad_accu[key]+=grad_tmp[key]
    for key in grad_accu.keys():
        grad_accu[key]/=batch_size
    return grad_accu


for epoch_ in range(epoch_num):
    print('******running epoch {}/{}********'.format(current_epoch,epoch_num))
    for i in range(data.train_num//batch_size):
        if i%100==99:
            print('running batch {}/{}'.format(i+1,data.train_num//batch_size))
        grad_tmp=train_batch(i,parameters)
        parameters=combine_parameters(parameters,grad_tmp,learn_rate)
    current_epoch+=1
    valid_log(parameters,data)
#################################################################

f = open(PATH/'mnist_np_model.pkl', 'wb')
pickle.dump(parameters, f)
f.close()

show_acc_log()
show_loss_log()