import numpy as np


class DataLoader():
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        xy = np.loadtxt('csv_data/answers.csv', delimiter=',')

        self.global_min = np.min(xy, 0)
        numerator = xy - self.global_min
        # 최대 - 최소 계산
        self.denominator = np.max(xy, 0) - np.min(xy, 0)
        xy = numerator / (self.denominator + 1e-7)

        self.x = xy
        self.y = xy[:, [-1]]  # Close as label

        # build a dataset
        data_x = []
        data_y = []
        for i in range(0, len(self.y) - FLAGS.seq_length):
            _x = self.x[i:i + FLAGS.seq_length]
            _y = self.y[i + FLAGS.seq_length]  # Next close price
            # print(_x, "->", _y)
            data_x.append(_x)
            data_y.append(_y)

        # train/test split
        train_size = int(len(data_y) * 0.7)
        self.trainX, self.testX = np.array(data_x[0:train_size]), np.array(
            data_x[train_size:len(data_x)])
        self.trainY, self.testY = np.array(data_y[0:train_size]), np.array(
            data_y[train_size:len(data_y)])

    def get_train(self):
        return self.trainX, self.trainY

    def get_test(self):
        return self.testX, self.testY

    def get_last(self):
        return np.array([self.x[-self.FLAGS.seq_length:]])

    def last_data_process(self, l_pred):
        prev_data = (self.y * (self.denominator + 1e-7) + self.global_min)[-1, -1]
        pred_data = (l_pred * (self.denominator + 1e-7) + self.global_min)[:, [-1]][0][0]

        print("""
            prev : {}, prediction: {}
            how {}
            """.format(prev_data, pred_data, prev_data > pred_data))
