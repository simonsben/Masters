from utilities.data_management import make_path, check_existence, open_w_dask, check_writable, make_dir
from matplotlib.pyplot import show
from utilities.plotting import hist_plot
from config import dataset

# Define paths
figure_path = make_path('figures/') / dataset / 'analysis' / 'tokens_per_document.png'
dataset_path = make_path('data/prepared_data/') / (dataset + '.csv')

# Ensure paths are valid
check_existence(dataset_path)
make_dir(figure_path)

# Load data
documents = open_w_dask(dataset_path, dtypes={'hyperlinks': 'object'})['document_content'].astype(str)
print('Content loaded')

# Compute word counts
document_lengths = documents.map_partitions(
    lambda df: df.apply(lambda document: len(document.split(' '))),
    meta=int
).compute().values

# Plot histogram of word counts
axis_labels = ('Document word count', 'Number of documents')
hist_plot(document_lengths, 'Histogram of document word counts', figure_path, axis_labels)

show()
