# 加载训练的模型
import numpy as np
import keras
import pandas as pd
from keras import layers
from matplotlib import pyplot as plt
from keras.datasets import mnist as mn
from keras.models import load_model

model = load_model("keras_mnist.h5")

(train_img, train_lab), (test_img, test_lab) = mn.load_data()

result = model.predict(test_img)
def show_test(index):
    plt.imsave("c.png",test_img[index],cmap='gray')
    print("图片内容 : {}".format(test_lab[index]))
    print("预测 : {}".format(result[index].argmax()))
    
index = np.random.randint(1, len(test_img))
show_test(index)