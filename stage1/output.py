# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import math

'''    
제출용 코드
'''

f = open('output2.txt', 'w', encoding='utf-8', newline='')

# Instantiates a client
client = language.LanguageServiceClient()

data_file = open('k/grading.input', 'r', encoding='utf-8', newline='')

lines = data_file.readlines()

for line in lines:
    try:
        document = types.Document(
            content=line,
            type=enums.Document.Type.PLAIN_TEXT)

        # Detects the sentiment of the text
        sentiment = client.analyze_sentiment(document=document).document_sentiment

    except:
        pass

    value = math.exp(sentiment.score)
    value2 = 'NEU'

    # 엑셀로 4000개 데이터 평균 낸 걸로 정했다
    if value < 0.9:
        value2 = 'NEG'
    elif value > 1.68:
        value2 = 'POS'

    print('{}, {}, {}'.format(line, value, value2))
    f.write(value2 + '\n')

f.close()
data_file.close()
