import numpy as np
import pickle
from oper_data import TrainData
# import matplotlib.pyplot as plt
from nn_fun import predict
from pathlib import Path
PATH = Path('./model')
SHOWPATH= Path('./show')
###############################导入数据
data = TrainData()
data.open_file()
################################


f = open(PATH/'mnist_np_model.pkl', 'rb')
param = pickle.load(f)
# print(param)
f.close

test_index = np.random.randint(1000)
data.show_test(SHOWPATH,test_index)
predict_result = predict(data.test_img[test_index], param)
print("predict:{}".format(predict_result.argmax()))


