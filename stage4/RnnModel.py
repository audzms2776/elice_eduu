import tensorflow as tf
from tensorflow.contrib import rnn

tf.set_random_seed(777)  # reproducibility


class RnnModel:
    def __init__(self, sess, FLAGS):
        self.FLAGS = FLAGS
        self.name = FLAGS.model_name
        self.sess = sess
        self._build_net()

    def _build_net(self):
        self.X = tf.placeholder(tf.float32, [None, self.FLAGS.seq_length, self.FLAGS.data_dim])
        self.Y = tf.placeholder(tf.float32, [None, 1])

        # build a LSTM network
        cell = rnn.BasicLSTMCell(num_units=self.FLAGS.hidden_dim, state_is_tuple=True, activation=tf.tanh)
        # cell = rnn.MultiRNNCell(
        #     [rnn.BasicLSTMCell(self.FLAGS.hidden_dim, state_is_tuple=True, activation=tf.tanh) for _ in range(10)],
        #     state_is_tuple=True)

        outputs, _states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)

        curr_layer = tf.reshape(outputs[:, -1], [-1, 5, 5, 1])
        conv1 = tf.layers.conv2d(inputs=curr_layer, filters=10, kernel_size=[3, 3], activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=20, kernel_size=[2, 2], activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=1, kernel_size=[2, 2], activation=None)

        self.Y_pred = tf.reshape(conv3, (-1, 1))

        # cost/loss
        self.loss = tf.losses.mean_squared_error(self.Y, self.Y_pred)
        # tf.summary.scalar('cost', self.loss)
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate).minimize(self.loss)

    def train(self, train_x, train_y):
        return self.sess.run([self.optimizer, self.loss], feed_dict={
            self.X: train_x, self.Y: train_y
        })

    def test(self, test_x):
        return self.sess.run([self.Y_pred], feed_dict={
            self.X: test_x
        })
