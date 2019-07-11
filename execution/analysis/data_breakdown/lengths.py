from utilities.data_management import make_path, check_existence, move_to_root, open_w_dask
from numpy import min, max, mean, std
from matplotlib.pyplot import show, figure

move_to_root(4)

dataset_name = 'storm-front'
dataset_path = make_path('data/prepared_data/') / (dataset_name + '.csv')

check_existence(dataset_path)

dataset = open_w_dask(dataset_path, dtypes={'hyperlinks': 'object'})
content = dataset['document_content'].astype(str)
print('Content loaded')

document_lengths = content.map_partitions(
    lambda df: df.apply(lambda document: len(document.split(' '))),
    meta=int
).compute()

print('word length desc')
print(document_lengths.describe())

figure()
ax = document_lengths.hist(bins=25, log=True)

ax.set_title('Document word counts')
ax.set_xlabel('Number of words')
ax.set_ylabel('Number of documents')


def lengths(document, metric):
    word_lengths = []

    for word in document.split(' '):
        length = len(word)
        if length > 0:
            word_lengths.append(length)

    if len(word_lengths) < 1:
        return 0
    return metric(word_lengths)


def apply_metric(documents, metric):
    return documents.map_partitions(
        lambda df: df.apply(
            lambda document: lengths(document, metric)
        ),
        meta=int
    ).compute()


vals = apply_metric(content, max)

figure()
ax = vals.hist(bins=25, log=True)

ax.set_title('Max word lengths')
ax.set_xlabel('Word length')
ax.set_ylabel('Number of words')

show()
