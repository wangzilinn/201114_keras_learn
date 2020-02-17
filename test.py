from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()
test = X_train[18]
final = test[np.newaxis, :]
model = load_model('h5/RNN_classifier.h5')

result = model.predict(X_train[:1])

print(result)

fig = plt.figure()
ax = fig.add_subplot(111)
data = X_train[1]
im = ax.imshow(data, cmap='viridis')
plt.show()
