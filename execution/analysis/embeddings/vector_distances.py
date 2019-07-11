from dask.dataframe import read_csv
from utilities.data_management import load_execution_params, make_path, move_to_root, check_existence, make_dir
from matplotlib.pyplot import show, savefig, tight_layout
from utilities.analysis import get_nearest_neighbours, get_relative_neighbours, svd_embeddings
from utilities.plotting import scatter_plot, plot_embedding_rep, scatter_3_plot
from scipy.linalg import norm as two_norm
# from scipy.stats import pearsonr
# from numpy import abs, argsort


# Define parameters
target_word = 'good'
targets = ('good', 'bad', 'poor')
max_cos_dist = .85

# Define paths
move_to_root(4)
params = load_execution_params()
embed_name = params['fast_text_model']
data_name = params['dataset']
embed_path = make_path('data/prepared_lexicon/') / (embed_name + '.csv')
dest_dir = make_path('data/processed_data/') / data_name / 'analysis' / 'embedding_neighbours' / target_word

check_existence(embed_path)
make_dir(dest_dir, 3)


# Define dataset-specific constants
dtypes = {str(ind): float for ind in range(1, 301)}
dtypes[0] = str

# Import data
embeddings = read_csv(embed_path, dtype=dtypes)
print('Data imported')

embeddings = svd_embeddings(embeddings)
print('Embeddings ready, calculating nearest neighbours')

words, target = get_nearest_neighbours(embeddings, target_word, n_words=250)
# words, target = get_relative_neighbours(embeddings, targets, max_angle=max_cos_dist, n_words=250)
print(words)
words.to_csv(dest_dir / 'data.csv')

# Plot histogram of metrics
metrics = ['euclidean_distances', 'cosine_distances']
[axes] = words.hist(column=metrics, bins=40, figsize=(8, 5))

# Add titles to histograms
for ax, metric in zip(axes, metrics):
    ax.set_xlabel(metric.replace('_', ' ').capitalize())
    ax.set_ylabel('Number of vectors')
    ax.set_title('Histogram of distances from ' + target_word)
savefig(dest_dir / 'metric_histogram.png')

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
savefig(dest_dir / 'metrics.png')

# Plot representation of source vector space
plot_embedding_rep(target_norm, words['euclidean_distances'].max(), words['cosine_distances'].max())
savefig(dest_dir / 'embedding_rep.png')

# Plot columns correlations
# columns = [str(ind) for ind in range(1, 51)]
# vectors = words[columns]
# correlations = abs([pearsonr(words['euclidean_distances'], vector)[0] for vector in vectors.values.transpose()])
# scatter_plot((list(range(len(correlations))), correlations), 'Dimension correlations for ' + target_word)
#
# # Calculate top columns for neighbours plot
# top_columns = list(argsort(correlations)[-3:].astype(str))
# print('Top columns', top_columns)
top_columns = ['1', '2', '3']

# Plot neighbours using the first 3 singular value dimensions
vectors = words[top_columns].values[1:].transpose()
axis_titles = ['Dimension ' + str(ind) for ind in range(1, 4)]
ax = scatter_3_plot(vectors, 'Neighbour embeddings for ' + target_word, words['euclidean_distances'].values[1:], ax_titles=axis_titles,
                    c_bar_title='Euclidean distance', cmap='Blues_r')

# Plot target point
x, y, z = words.iloc[:1][top_columns].values.transpose()
ax.scatter(x, y, z, c='r', s=50, edgecolors='k')

tight_layout()
savefig(dest_dir / 'word_cloud.png')

show()
