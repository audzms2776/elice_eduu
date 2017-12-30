'''
This script shows how to predict stock prices using a basic RNN
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn

tf.set_random_seed(777)  # reproducibility

# train Parameters
seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 2000

# Open, High, Low, Volume, Close
# xy = np.loadtxt('stock.csv', delimiter=',')
xy = np.loadtxt('csv_data/answers.csv', delimiter=',')

global_min = np.min(xy, 0)
numerator = xy - global_min
# 최대 - 최소 계산
denominator = np.max(xy, 0) - np.min(xy, 0)
xy = numerator / (denominator + 1e-7)

x = xy
y = xy[:, [-1]]  # Close as label

# build a dataset
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    # print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# train/test split
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
cell = rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.layers.dense(outputs[:, -1], output_dim, activation=None)  # We use the last cell's output

# cost/loss
loss = tf.losses.mean_squared_error(Y, Y_pred)
# optimizer
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
            X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    # Plot predictions
    plt.plot(testY)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()

    l_data = np.array([x[-seq_length:]])
    l_pred = sess.run(Y_pred, feed_dict={
        X: l_data
    })

    prev_data = (y * (denominator + 1e-7) + global_min)[-1, -1]
    pred_data = (l_pred * (denominator + 1e-7) + global_min)[:, [-1]][0][0]

    print("""
    prev : {}, prediction: {}
    how {}
    """.format(prev_data, pred_data, prev_data > pred_data))
