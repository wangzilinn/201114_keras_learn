from keras import Input, Model
from keras.datasets import mnist
from keras.layers import Dense
import matplotlib.pyplot as plt

(x_train, _), (x_test, y_test) = mnist.load_data()

# pre-processing
X_train = x_train.astype('float32') / 255. - 0.5
X_test = x_test.astype('float32') / 255. - 0.5
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

encoding_dim = 2

input_img = Input(shape=(784, ))

# encoder layers
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoded_output = Dense(encoding_dim)(encoded)  # 注意这里没有激活函数

# decoder layers
decoded = Dense(10, activation='relu')(encoded_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)  # 注意这里使用tanh

# construct the autoencoder model
autoencoder = Model(input=input_img, output=decoded)
# construct the encoder
encoder = Model(input=input_img, output=encoded_output)

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# training
autoencoder.fit(X_train, X_train, epochs=20, batch_size=256, shuffle=True)

# plotting
encoded_img = encoder.predict(X_test)
plt.scatter(encoded_img[:, 0], encoded_img[:, 1], c=y_test)
plt.colorbar()
plt.show()
