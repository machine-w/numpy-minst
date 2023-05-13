import math
import numpy as np
import copy
from pathlib import Path
import struct
import pickle
import matplotlib.pyplot as plt
PATH ='./imgs/'
###############################导入数据
dataset_path=Path('./MNIST')
train_img_path=dataset_path/'train-images.idx3-ubyte'
train_lab_path=dataset_path/'train-labels.idx1-ubyte'
test_img_path=dataset_path/'t10k-images.idx3-ubyte'
test_lab_path=dataset_path/'t10k-labels.idx1-ubyte'


class TrainData():
    train_num=50000
    valid_num=10000
    test_num=10000
    def open_file(self):
        with open(train_img_path,'rb') as f:
            struct.unpack('>4i',f.read(16))
            tmp_img=np.fromfile(f,dtype=np.uint8).reshape(-1,28*28)/255
            self.train_img=tmp_img[:self.train_num]
            self.valid_img=tmp_img[self.train_num:]
            
        with open(test_img_path,'rb') as f:
            struct.unpack('>4i',f.read(16))
            self.test_img=np.fromfile(f,dtype=np.uint8).reshape(-1,28*28)/255

        with open(train_lab_path,'rb') as f:
            struct.unpack('>2i',f.read(8))
            tmp_lab=np.fromfile(f,dtype=np.uint8)
            self.train_lab=tmp_lab[:self.train_num]
            self.valid_lab=tmp_lab[self.train_num:]
            
        with open(test_lab_path,'rb') as f:
            struct.unpack('>2i',f.read(8))
            self.test_lab=np.fromfile(f,dtype=np.uint8)
    
    def show_train(self,path,index):
        plt.imsave(path/(str(index+1)+".png"),self.train_img[index].reshape(28,28),cmap='gray')
        print('label : {}'.format(self.train_lab[index]))
    def show_valid(self,path,index):
        plt.imsave(path/(str(index+1)+".png"),self.valid_img[index].reshape(28,28),cmap='gray')
        print('label : {}'.format(self.valid_lab[index]))
    def show_test(self,path,index):
        plt.imsave(path/(str(index+1)+".png"),self.test_img[index].reshape(28,28),cmap='gray')
        print('label : {}'.format(self.test_lab[index]))

    def save_train_lab(self):
        np.savetxt(Path(PATH+"train.csv"), self.train_lab.astype(int),fmt='%i', delimiter=",")
    def save_valid_lab(self):
        np.savetxt(Path(PATH+"valid.csv"), self.valid_lab.astype(int),fmt='%i', delimiter=",")
    def save_test_lab(self):
        np.savetxt(Path(PATH+"test.csv"), self.test_lab.astype(int),fmt='%i', delimiter=",")

if __name__ == '__main__':
    data = TrainData()
    data.open_file()
    for i in range(data.train_num):
        data.show_train(Path(PATH+"train"),i)
    for i in range(data.valid_num):
        data.show_valid(Path(PATH+"valid"),i)
    for i in range(data.test_num):
        data.show_test(Path(PATH+"test"),i)
    data.save_test_lab()
    data.save_train_lab()
    data.save_valid_lab()