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
        # cell = rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
        cell = rnn.MultiRNNCell(
            [rnn.BasicLSTMCell(self.FLAGS.hidden_dim, state_is_tuple=True, activation=tf.tanh) for _ in range(10)],
            state_is_tuple=True)

        outputs, _states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)
        layer1 = tf.layers.dense(outputs[:, -1], 30, activation=tf.nn.relu)
        layer2 = tf.layers.dense(layer1, 10, activation=tf.nn.relu)
        self.Y_pred = tf.layers.dense(layer2, self.FLAGS.output_dim, activation=None)  # We use the last cell's output

        # cost/loss
        self.loss = tf.losses.mean_squared_error(self.Y, self.Y_pred)
        tf.summary.scalar('cost', self.loss)
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate).minimize(self.loss)

    def train(self, train_x, train_y, summary):
        return self.sess.run([self.optimizer, self.loss, summary], feed_dict={
            self.X: train_x, self.Y: train_y
        })

    def test(self, test_x):
        return self.sess.run(self.Y_pred, feed_dict={
            self.X: test_x
        })
