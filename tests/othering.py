from utilities.data_management import open_w_pandas
from model.extraction import othering_vector

filename = '../data/prepared_data/24k-abusive-tweets.csv'

dataset = open_w_pandas(filename).iloc[:100]
print(dataset)

othering_vector(dataset)
