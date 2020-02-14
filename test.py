from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import load_model

(X_train, y_train), (X_test, y_test) = mnist.load_data()
test = X_train[18].reshape(1, -1)  # shape:(1,784)

model = load_model('h5/classifier.h5')

result = model.predict(test)

print(result)

fig = plt.figure()
ax = fig.add_subplot(111)
data = X_train[18]
im = ax.imshow(data, cmap='viridis')
plt.show()
