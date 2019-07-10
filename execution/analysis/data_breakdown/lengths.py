from utilities.data_management import make_path, check_existence, move_to_root, open_w_dask, open_w_pandas
from utilities.analysis import length_stats
from numpy import min, max, mean, std
# from matplotlib.pyplot import show
# from dask.array import histogram, from_array

move_to_root(4)

dataset_name = 'storm-front'
dataset_path = make_path('data/prepared_data/') / (dataset_name + '.csv')

check_existence(dataset_path)

dataset = open_w_pandas(dataset_path)
content = dataset['document_content'].values.astype(str)
print('Content loaded')


document_lengths = content.apply(lambda document: document.split(' '))
word_lengths = content.apply(lambda document: [len(word) for word in document])

print(document_lengths)
print(word_lengths)
