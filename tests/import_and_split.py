from utilities.data_management import split_sets, open_w_pandas
from execution.processing import splitter

filename = '../data/prepared_data/24k-abusive-tweets.csv'

dataset = open_w_pandas(filename)
print(dataset)

train, test = split_sets(dataset, splitter)

print(train)
print(test)
