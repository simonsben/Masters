from dask.dataframe import read_csv
from utilities.data_management import make_path, move_to_root, check_existence, make_dir
from matplotlib.pyplot import show, savefig, tight_layout, subplots
from utilities.analysis import svd_embeddings
from utilities.plotting import  scatter_3_plot
import config

# Define paths
move_to_root(4)
embed_name = config.fast_text_model
data_name = config.dataset

embed_path = make_path('data/prepared_lexicon/') / (embed_name + '.csv')
dest_dir = make_path('figures/') / data_name / 'analysis' / 'clouds'

check_existence(embed_path)
make_dir(dest_dir)


# Define dataset-specific constants
dtypes = {str(ind): float for ind in range(1, 301)}
dtypes[0] = str

# Import data
embeddings = read_csv(embed_path, dtype=dtypes)
print('Data imported')

# Calculate svd embeddings
embeddings = svd_embeddings(embeddings, dimensions=3).iloc[:, 1:].sample(frac=.25).values.transpose().compute()


# Plot embedding cloud
scatter_3_plot(embeddings, 'Embeddings point cloud', ax_titles=['Dimension ' + str(ind) for ind in range(1, 4)],
               size=5, filename=dest_dir / 'embedding_cloud.png')


# Plot histogram of embedding dimensions
fig, axes = subplots(3, sharex='col', sharey='col')
axes[-1].set_xlabel('Embedding value')

for ind, ax in enumerate(axes):
    ax.hist(embeddings[ind], bins=35)
    ax.set_title('Embedding histogram for dimension ' + str(ind + 1))

tight_layout()
savefig(dest_dir / 'histograms.png')

show()
