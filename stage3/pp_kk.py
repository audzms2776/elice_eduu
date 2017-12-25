from multiprocessing import Process, Queue
import json
from functools import reduce
from konlpy.tag import Kkma

kkma = Kkma()
queue = Queue()

f = open('words.txt', 'w', encoding='utf-8')
pubs = ['JTBC', 'KBS 뉴스', 'MBN', 'TV조선', 'YTN', '경향신문', '국민일보', '노컷뉴스', '뉴시스', '동아일보', '매일경제',
        '서울신문', '아시아경제', '연합뉴스', '연합뉴스TV', '오마이뉴스', '채널A', '파이낸셜뉴스', '헤럴드경제']


def replace_text(s):
    return s.replace('\r', '') \
        .replace('\n', '') \
        .replace('    ', '') \
        .replace('[<언론사명>tv 제공]', '') \
        .replace('[<언론사명> 자료사진]', '') \
        .replace('<언론사명>', '') \
        .replace('\'', '') \
        .replace('\"', '')


def creator(arr, q):
    for idx, item in enumerate(arr):
        q.put({
            'text': item['text'],
            'pub': pubs.index(item['writing']),
            'idx': idx
        })


def my_consumer(q):
    while True:
        get_data = q.get()
        print(get_data['idx'])

        nn = replace_text(get_data['text'])
        kk_nn = kkma.nouns(nn)

        # print(kk_nn)
        nouns = '{},{}\n'.format(get_data['pub'], reduce(lambda x, y: '{},{}'.format(x, y), kk_nn))
        # print(nouns)

        f.write(nouns)

        # if get_data == 9:
        #     print('close !')
        #     close_queue()


def close_queue():
    queue.close()
    queue.join_thread()


if __name__ == '__main__':
    with open('D:\\0_School_Hannam\\elice\\stage3\\elice\\data\\train_data.json') as data_file:
        # with open('D:\\0_School_Hannam\\elice\\stage3\\elice\\data\\test.input.json') as data_file:
        data = json.load(data_file)

    # data = range(10)
    ps_one = Process(target=creator, args=(data, queue))
    ps_two = [Process(target=my_consumer, args=(queue,)) for _ in range(5)]

    ps_one.start()
    list(map(lambda x: x.start(), ps_two))

    # ps_one.join()
    # list(map(lambda x: x.join(), ps_two))
