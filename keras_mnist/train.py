import numpy as np
import keras
import pandas as pd
from keras import layers
from matplotlib import pyplot as plt
from keras.datasets import mnist as mn


# 读取训练数据和测试数据
(train_img, train_lab), (test_img, test_lab) = mn.load_data()
model = keras.Sequential()
model.add(layers.Flatten()) # (60000, 28, 28) => (60000, 28*28)
model.add(layers.Dense(64, activation='tanh'))
model.add(layers.Dense(10, activation='softmax'))

# 编译模型
model.compile(
    optimizer="adam",
    # 注意因为label是顺序编码的,这里用这个
    loss='sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

# 模型结构
# model.summary()

# 使用history保存每个epoch结束的loss，accuracy等信息
history = model.fit(train_img, train_lab, epochs=10, batch_size=500, validation_data=(test_img, test_lab), verbose=2) # 每批500张图片

plt.plot(history.history['val_accuracy'], c='g', label='validation acc')
plt.plot(history.history['accuracy'], c='b', label='train acc')
plt.legend()
plt.savefig("a.png")

plt.plot(history.history['val_loss'], c='g', label='validation loss')
plt.plot(history.history['loss'], c='b', label='train loss')
plt.legend()
plt.savefig("b.png")



# 保存模型
model.save('keras_mnist.h5')