'''
This script shows how to predict stock prices using a basic RNN
'''
# import tensorflow as tf
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
import os


def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# train Parameters
seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 1000

# Open, High, Low, Volume, Close
# xy = np.loadtxt('stock.csv', delimiter=',')
xy = np.loadtxt('stock.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)
xy = MinMaxScaler(xy)
x = xy  # 데이터 전부
y = xy[:, [-1]]  # Close as label 종가 데이터

# build a dataset
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    # print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

print(x[-seq_length:])

# train/test split
# train_size = int(len(dataY) * 0.7)
# test_size = len(dataY) - train_size
# trainX, testX = np.array(dataX[0:train_size]), np.array(
#     dataX[train_size:len(dataX)])
# trainY, testY = np.array(dataY[0:train_size]), np.array(
#     dataY[train_size:len(dataY)])
