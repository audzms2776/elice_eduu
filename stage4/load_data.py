import json
import pprint
import csv

name_dict = dict()


def create_key(x):
    name_dict[x] = []


def put_data(x):
    com_name = x[1]
    name_dict[com_name].append(x[2:])


with open('data/train_data.json') as data_file:
    data = json.load(data_file)

names = list(set([x[1] for x in data]))
print(len(names))
list(map(create_key, names))
list(map(put_data, data))

pprint.pprint(name_dict['teller'])

for name in names:
    f = open('csv_data/{}.csv'.format(name), 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerows(name_dict[name])
    f.close()
