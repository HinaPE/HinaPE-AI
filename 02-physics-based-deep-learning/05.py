# Pure supervised learning

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

N = 200  # number of data points
X = np.random.random(N)
sign = np.random.choice([-1, 1], size=N)
Y = np.sqrt(X) * sign

act = tf.keras.layers.ReLU()
nn_sv = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(10, activation=act),
    tf.keras.layers.Dense(10, activation=act),
    tf.keras.layers.Dense(1, activation='linear')])

loss_sv = tf.keras.losses.MeanSquaredError()
optimizer_sv = tf.keras.optimizers.Adam(learning_rate=0.01)
nn_sv.compile(optimizer=optimizer_sv, loss=loss_sv)

results_sv = nn_sv.fit(X, Y, epochs=1000, verbose=0)

# Results
plt.plot(X, Y, '.', label='Data points', color="lightgray")
plt.plot(X, nn_sv.predict(X), '.', label='Supervised', color="red")
plt.xlabel('y')
plt.ylabel('x')
plt.title('Standard approach')
plt.legend()
plt.show()

nn_dp = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(10, activation=act),
    tf.keras.layers.Dense(10, activation=act),
    tf.keras.layers.Dense(1, activation='linear')])

loss_dp = tf.keras.losses.MeanSquaredError()
nn_dp.compile(optimizer=optimizer_sv, loss=loss_sv)
