import numpy as np
from keras.datasets import mnist
'''classifier'''
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
#
# data = (X_train[18] / 255).reshape(1, -1).tolist()[0]  # 输出1行数据
#
# # 格式化数据, 用keil使用:
# cnt = 0
# for i in range(28):
#     for j in range(28):
#         print("%.4f" % data[cnt], end=", ")
#         cnt = cnt + 1
#     print("\n")

'''LSTM'''
xs = np.arange(0, 20) / (10 * np.pi)
for i in range(20):
    print("%.4f" % xs[i], end=", ")
