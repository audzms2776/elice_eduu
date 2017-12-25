import tensorflow as tf
import numpy as np
from numpy import genfromtxt

test_data = genfromtxt('test.csv', delimiter=',', dtype=np.float32)
n_test_data = test_data.reshape((-1, 3, 7, 5))

training_epochs = 30
learning_rate = 0.001
# num_steps = 2000
batch_size = 500
dropout = 0.25  # Dropout, probability to drop a unit


def make_sub(arr):
    f = open("submission.txt", 'w', encoding='utf-8', newline='')
    for d in arr:
        team = 0

        if d[0] + d[1] < d[2] + d[3]:
            team = 100
        else:
            team = 200
        print(team)
        f.write(str(team) + '\n')
    f.close()


def cnn_net(x, dropout, is_training):
    with tf.variable_scope('ConvNet'):
        conv1 = tf.layers.conv2d(x, 20, 1, activation=tf.nn.relu, padding='SAME')
        conv2 = tf.layers.conv2d(conv1, 30, 3, activation=tf.nn.relu, padding='SAME')
        conv3 = tf.layers.conv2d(conv2, 40, 2, activation=tf.nn.relu, padding='SAME')
        conv4 = tf.layers.conv2d(conv3, 50, 2, activation=tf.nn.relu, padding='SAME')
        conv5 = tf.layers.conv2d(conv4, 60, 2, activation=tf.nn.relu)

        fc1 = tf.contrib.layers.flatten(conv5)
        fc1 = tf.layers.dense(fc1, 200, activation=tf.nn.relu)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        fc2 = tf.layers.dense(fc1, 100, activation=tf.nn.relu)
        fc2 = tf.layers.dropout(fc2, rate=dropout, training=is_training)

        out = tf.layers.dense(fc2, 2)
    return out


def model_fn():
    xx = tf.placeholder(tf.float32, [None, 3, 7, 5])
    logits_test = cnn_net(xx, dropout, is_training=False)

    result = tf.reshape(logits_test, (-1, 4))

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, '/tmp/model2/model.ckpt')

    total_batch = int(16000 / batch_size)
    total_result = []

    for i in range(total_batch):
        batch_x = n_test_data[i * batch_size: (i + 1) * batch_size]

        prediction = sess.run(result, feed_dict={
            xx: batch_x
        })

        total_result += list(prediction)
        print(prediction)

    print(len(total_result))
    make_sub(total_result)


if __name__ == '__main__':
    model_fn()
