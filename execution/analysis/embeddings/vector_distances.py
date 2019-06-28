from dask.dataframe import read_csv
from utilities.data_management import load_execution_params, make_path, move_to_root, check_existence
from matplotlib.pyplot import show
from utilities.analysis import get_nearest_neighbours, get_relative_neighbours, embeddings_to_svd
from utilities.plotting import scatter_plot, plot_embedding_rep
from scipy.linalg import norm as two_norm

# Define paths
move_to_root(4)
embed_name = load_execution_params()['fast_text_model']
embed_path = make_path('data/prepared_lexicon/') / (embed_name + '.csv')
check_existence(embed_path)

# Define dataset-specific constants
dtypes = {str(ind): float for ind in range(1, 301)}
dtypes[0] = str

# Define parameters
target_word = 'fuck'
targets = ('good', 'bad', 'poor')
max_cos_dist = .85

# Import data
embeddings = read_csv(embed_path, dtype=dtypes)
print('Data imported')

embeddings = embeddings_to_svd(embeddings)
print('Embeddings ready, calculating nearest neighbours')

words, target = get_nearest_neighbours(embeddings, target_word, n_words=250)
# words, target = get_relative_neighbours(embeddings, targets, max_angle=max_cos_dist, n_words=250)
print(words)

# Plot histogram of metrics
metrics = ['euclidean_distances', 'cosine_distances']
[axes] = words.hist(column=metrics, bins=40, figsize=(8, 5))

# Add titles to histograms
for ax, metric in zip(axes, metrics):
    ax.set_xlabel(metric.replace('_', ' ').capitalize())
    ax.set_ylabel('Number of vectors')
    ax.set_title('Histogram of distances from ' + target_word)

# Plot scatter of metric relationship
x_key, y_key, weight_key = 'cosine_distances', 'euclidean_distances', 'vector_norms'
target_norm = two_norm(target)

# Plot scatter metrics
ax = scatter_plot((words[x_key], words[y_key]), 'Metric relationship for ' + target_word, words[weight_key],
                  c_bar_title=weight_key.replace('_', ' ').capitalize())
ax.set_xlabel(x_key.replace('_', ' ').capitalize())
ax.set_ylabel(y_key.replace('_', ' ').capitalize())

ax.scatter(0, 0, c='g', s=50)
ax.legend(['Similar words', 'Target word'], loc='lower right')

# Plot representation of source vector space
plot_embedding_rep(target_norm, words['euclidean_distances'].max(), max_cos_dist)

show()
