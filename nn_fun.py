#定义需要的两个激活函数
import math
import numpy as np
import copy
from pathlib import Path
import struct


def tanh(x): 
    return np.tanh(x) # (e^x - e^-x) /(e^x + e^-x)
def softmax(x):
    exp=np.exp(x-x.max())
    return exp/exp.sum()
#定义两个激活函数的导数
def d_softmax(data):
    sm=softmax(data)
    return np.diag(sm)-np.outer(sm,sm)
def d_tanh(data):
    return 1/(np.cosh(data))**2

def predict(img,parameters):
    l0_in=img+parameters[0]['b']
    l0_out=tanh(l0_in)
    l1_in=np.dot(l0_out,parameters[1]['w'])+parameters[1]['b']
    l1_out=softmax(l1_in)
    return l1_out