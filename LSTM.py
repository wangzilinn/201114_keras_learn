import numpy as np
from keras import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam

# np.random.seed(1996)
import matplotlib.pyplot as plt

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.06


def get_batch():
    global BATCH_START, TIME_STEPS
    # arange(BATCH_START, BATCH_START + BATCH_SIZE * TIME_STEPS) 生成长度1000的等差数列
    # reshape((BATCH_SIZE, TIME_STEPS)) 修改为BATCH_SIZE(50)行, 每行TIME_STEPS(20)个的矩阵
    # 一行是一批, 一批有20个时间点, 一个方括号存多个数
    xs = np.arange(BATCH_START, BATCH_START + BATCH_SIZE * TIME_STEPS).reshape((BATCH_SIZE, TIME_STEPS)) / (10 * np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


model = Sequential()
model.add(LSTM(
    # batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),
    input_shape=(TIME_STEPS, INPUT_SIZE),
    output_dim=CELL_SIZE,
    return_sequences=True,  # output at all steps
    # stateful=True,  # 每批之间保存状态
))
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
adam = Adam(LR)
model.compile(optimizer=adam, loss='mse')

print("training...")
X_batch, Y_batch, xs = get_batch()

for step in range(501):
    X_batch, Y_batch, xs = get_batch()
    cost = model.train_on_batch(X_batch, Y_batch)
    pred = model.predict(X_batch, BATCH_SIZE)
    # print(X_batch.shape)
    # print(pred.shape)
    # plot:
    plt.plot(xs[0, :], Y_batch[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.01)
    if step % 10 == 0:
        print('train cost: ', cost)
plt.show()
model.save("h5/LSTM.h5")
