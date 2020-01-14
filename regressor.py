import numpy as np
np.random.seed(1996)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))

# plot data
plt.scatter(X, Y)
plt.show()

# define data
X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]
# 定义模型
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))
# 定义模型的执行方式
model.compile(loss="mse", optimizer="sgd")  # mes:均方误差, sgd:随机梯度下降

# training
print("training---------------")
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print("cost: ", cost)

# test
print("\nTesting")
cost = model.evaluate(X_test, Y_test, batch_size=40)
print("test cost: ", cost)
W, b = model.layers[0].get_weights()
print("weights:", W, "biases:", b)

# plotting the prediction
# Y_pred = model.predict(X_test)
# plt.scatter(X_test, Y_test)
# plt.plot(X_test, Y_pred)
# plt.show()






