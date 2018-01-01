import tensorflow as tf
from RnnModel import RnnModel
from DataLoad import DataLoader
import matplotlib.pyplot as plt
import os

flag = tf.app.flags
flag.DEFINE_integer('iterations', 1000, 'iterations')
flag.DEFINE_integer('seq_length', 10, 'seq_length')
flag.DEFINE_string('csv_path', 'csv_data', 'csv data path')
flag.DEFINE_string('model_name', 'rnn', 'rnn model name')
flag.DEFINE_float('data_percent', 0.8, 'train : test ratio')
FLAGS = flag.FLAGS

f = open('submission.txt', 'w', encoding='utf-8', newline='\n')


def visual_graph(test_y, test_predict):
    plt.plot(test_y)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()


def train_fn(sess, graph, loader, name):
    sess.run(tf.global_variables_initializer())
    train_x, train_y = loader.get_train()

    for i in range(1, FLAGS.iterations + 1):
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('log/rnn-{}'.format(name))
        _, step_loss, summary = graph.train(train_x, train_y, merged_summary)

        if i % 50 == 0:
            writer.add_summary(summary, global_step=i)

        if i % 100 == 0:
            print("[step: {}] loss: {}".format(i, step_loss))


def test_fn(graph, loader):
    test_x, test_y = loader.get_test()
    test_prediction = graph.test(test_x)
    visual_graph(test_y, test_prediction)


def prediction_stock(graph, loader):
    # test_fn(graph, loader)

    last_data = loader.get_last()
    last_prediction = graph.test(last_data)
    prediction, direction = loader.last_data_process(last_prediction)

    return prediction, direction


def write_file(value, direction):
    f.write('{} {}\n'.format(direction, value))


def read_test_list():
    lines = open('data/grading.input.txt', 'r', encoding='utf-8', newline='')
    test_names = list(map(lambda x: x[:-1], lines))
    lines.close()

    return test_names


def main(_):
    sess = tf.Session()
    graph = RnnModel(sess, FLAGS.model_name)
    files = read_test_list()

    for idx, name in enumerate(files[:10]):
        print('{} :: {}/{}'.format(name, idx, len(files)))
        data_loader = DataLoader(os.path.join(FLAGS.csv_path, name) + '.csv', FLAGS)
        train_fn(sess, graph, data_loader, name)
        v, d = prediction_stock(graph, data_loader)
        write_file(v, d)

    f.close()


if __name__ == '__main__':
    tf.app.run()
