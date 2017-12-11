import json
# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import csv
import math
'''    
train_data.json 읽어서 google cloud language로 결과를 받아옴

<파일 형식>
e*x, rating_class

multi process로 바꾸자
'''

f = open('output', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)

# Instantiates a client
client = language.LanguageServiceClient()

data_file = open('data/train_data.json', encoding='utf-8')
lines = json.load(data_file)[:20]

for data in lines:
    document = types.Document(
        content=data['review'],
        type=enums.Document.Type.PLAIN_TEXT)

    # Detects the sentiment of the text
    sentiment = client.analyze_sentiment(document=document).document_sentiment

    value = math.exp(sentiment.score)

    rating_class = None

    if data['rating'] <= 3:
        rating_class = 1
    elif data['rating'] >= 8:
        rating_class = 3
    else:
        rating_class = 2

    wr.writerow([sentiment, rating_class])

f.close()
data_file.close()
