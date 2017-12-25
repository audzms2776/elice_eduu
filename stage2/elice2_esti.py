import tensorflow as tf
from numpy import genfromtxt
import numpy as np

train_data = genfromtxt('train.csv', delimiter=',')
label_data = genfromtxt('label2.csv', delimiter=',', dtype=np.int32)

n_train_data = train_data.reshape((-1, 3, 7, 5))
label_data = label_data.reshape((-1, 1))

print('###########\nload dataset!\n')

training_epochs = 300
learning_rate = 0.001
# num_steps = 2000
batch_size = 800
dropout = 0.25  # Dropout, probability to drop a unit


def cnn_net(x, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
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
    yy = tf.placeholder(tf.int32, [None, 2])

    logits_train = cnn_net(xx, dropout, reuse=False, is_training=True)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(yy, dtype=tf.float32),
                                                                     logits=logits_train))
    tf.summary.scalar('cost', loss_op)

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        loss=loss_op,
        global_step=tf.train.get_global_step()
    )

    sess = tf.Session()
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('elice_clasifi2')
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    step = 0

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(77358 / batch_size)

        for i in range(total_batch):
            batch_x = n_train_data[i * batch_size: (i + 1) * batch_size]
            batch_y = label_data[i * batch_size: (i + 1) * batch_size]
            hotdata_y = np.array(np.eye(2)[batch_y].reshape((-1, 2)))

            summary, c, _ = sess.run([merged_summary, loss_op, train_op], feed_dict={
                xx: batch_x,
                yy: hotdata_y
            })

            writer.add_summary(summary, global_step=step)
            step += 1

            avg_cost += c / total_batch
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

        if (epoch + 1) % 10 == 0:
            save_path = saver.save(sess, '/tmp/model2/model.ckpt')
            print(save_path)
    print('Learning Finished!')


def main():
    model_fn()


if __name__ == '__main__':
    main()
