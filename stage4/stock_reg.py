import tensorflow as tf
from RnnModel import RnnModel
from DataLoad import DataLoader
import matplotlib.pyplot as plt

flag = tf.app.flags
flag.DEFINE_integer('iterations', 2000, 'iterations')
flag.DEFINE_integer('seq_length', 7, 'seq_length')
FLAGS = flag.FLAGS


def visual_graph(test_y, test_predict):
    plt.plot(test_y)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()


def main(_):
    sess = tf.Session()
    graph = RnnModel(sess, 'rnn')
    data_loader = DataLoader(FLAGS)
    sess.run(tf.global_variables_initializer())
    train_x, train_y = data_loader.get_train()

    for i in range(FLAGS.iterations):
        _, step_loss = graph.train(train_x, train_y)
        print("[step: {}] loss: {}".format(i, step_loss))

    test_x, test_y = data_loader.get_test()
    test_prediction = graph.test(test_x)
    visual_graph(test_y, test_prediction)

    last_data = data_loader.get_last()
    last_prediction = graph.test(last_data)
    data_loader.last_data_process(last_prediction)


if __name__ == '__main__':
    tf.app.run()
