from multiprocessing import Process, Queue
import json
# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import math
import time


def creator(data, q):
    for idx, item in enumerate(data):
        q.put({
            'item': item,
            'idx': idx
        })


def my_consumer(q):
    client = language.LanguageServiceClient()

    while True:
        data = q.get()
        idx = data['idx']
        data = data['item']
        value = 0

        try:
            document = types.Document(
                content=data['review'],
                type=enums.Document.Type.PLAIN_TEXT)

            # Detects the sentiment of the text
            sentiment = client.analyze_sentiment(document=document).document_sentiment
            value = math.exp(sentiment.score)
        except Exception as e:
            print(e)
            pass

        rating_class = None

        if data['rating'] <= 3:
            rating_class = 1
        elif data['rating'] >= 8:
            rating_class = 3
        else:
            rating_class = 2

        time.sleep(2)
        print('{},{},{}'.format(idx, value, rating_class))


if __name__ == '__main__':
    json_file = open('k/train_data.json', encoding='utf-8')
    data = json.load(json_file)[:5000]

    ######################
    q = Queue()

    ps_one = Process(target=creator, args=(data, q))
    ps_two = [Process(target=my_consumer, args=(q,)) for _ in range(20)]

    ps_one.start()
    list(map(lambda x: x.start(), ps_two))

    q.close()
    q.join_thread()

    ps_one.join()
    list(map(lambda x: x.join(), ps_two))
