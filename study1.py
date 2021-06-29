import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from tensorflow import keras
from tensorflow.keras.layers import Dense

x = np.arange(-10, 10, 1)
y = 2 * x - 1

idx = np.arange(x.shape[0])
np.random.shuffle(idx)

x = x[idx]
y = y[idx]

plt.plot(x, y)
plt.show()

x_new = x.reshape(-1, 1)

lr = LinearRegression()
lr.fit(x_new, y)
print("기울기: ", lr.coef_)
print("y절편: ", lr.intercept_)

x_test = np.arange(11, 16, 1).reshape(-1, 1)
y_hat = lr.predict(x_test)

dnn = keras.Sequential()
dnn.add(Dense(units=1, input_shape=(1,)))
dnn.compile(optimizer="sgd", loss="mse")
dnn.summary()
dnn.fit(x_new, y, epochs=1000)

y_hat_dnn = dnn.predict(x_test)
print(y_hat_dnn)
