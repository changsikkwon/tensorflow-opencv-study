import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

data = pd.read_csv("dataset/samsung.csv")
data.head()

high_prices = data["고가"].values
low_prices = data["저가"].values
mid_prices = (high_prices + low_prices) / 2

seq_len = 50
sequence_length = seq_len + 1

result = [
    mid_prices[index : index + sequence_length]
    for index in range(len(mid_prices) - sequence_length)
]
normalized_data = [[((float(p) / float(window[0])) - 1) for p in window] for window in result]
result = np.array(normalized_data)

row = int(round(result.shape[0] * 0.9))
train = result[:row, :]
np.random.shuffle(train)

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1]

x_train.shape, x_test.shape

# build model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1, activation="linear"))
model.compile(loss="mse", optimizer="rmsprop")
model.summary()

# training
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=10, epochs=20)

pred = model.predict(x_test)

fig = plt.figure(facecolor="white", figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_test, label="True")
ax.plot(pred, label="Prediction")
ax.legend()
plt.show()
