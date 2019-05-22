from utilities.data_management import open_w_pandas
from model.extraction import othering_vector
from time import time

filename = '../data/prepared_data/24k-abusive-tweets.csv'

dataset = open_w_pandas(filename)

# 327s to execute -> 5.45 min
start = time()
document_matrix = othering_vector(dataset)
end = time()
print('Time to execute: ' + str(end - start))
print(document_matrix)
