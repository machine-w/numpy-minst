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

# test_index = np.random.randint(1000)
cmd = input("请输入测试图片号:")
if cmd == "all":
    success_num =0
    for i in range(1000):
        data.show_test(SHOWPATH,i)
        predict_result = predict(data.test_img[i], param)
        print("predict:{}".format(predict_result.argmax()))
        # print("########################",data.test_lab[i],predict_result.argmax())
        if str(data.test_lab[i]) == str(predict_result.argmax()):
            success_num +=1
    print("成功率:{}%".format(success_num/10))
else:
    try:
        test_index = int(cmd)
        data.show_test(SHOWPATH,test_index)
        predict_result = predict(data.test_img[test_index], param)
        print("predict:{}".format(predict_result.argmax()))
    except Exception as e:
        print("请输入对应数字或者all测试全部集合")




