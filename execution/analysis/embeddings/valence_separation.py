from utilities.data_management import move_to_root, make_path, load_tsv
from utilities.analysis import svd_embeddings
from dask.dataframe import read_csv
from numpy import asarray, vstack, min, max
from scipy.cluster.vq import whiten, kmeans
from scipy.spatial.distance import euclidean
from matplotlib.pyplot import subplots, show
from pandas import read_csv as panda_read
import config

move_to_root(4)

# Load execution parameters
dataset = config.dataset
lexicon = config.fast_text_model

# Define paths
source = make_path('data/prepared_lexicon/') / (dataset + '-' + lexicon + '.csv')
dictionary_path = make_path('data/lexicons/empath/dictionary.tsv')
svd_path = make_path('data/processed_data') / dataset / 'analysis' / 'embeddings' / 'svd_embeddings.csv'

# Import data
raw_dictionary = load_tsv(dictionary_path)
dictionary = {theme[0]: theme[1:] for theme in raw_dictionary}

if not svd_path.exists():
    raw_embeddings = read_csv(source)
    print('Data loaded')

    raw_vectors = svd_embeddings(raw_embeddings).compute()
    print('SVD vectors computed, saving')

    raw_vectors.to_csv(svd_path)
    print('Saved SVD vectors')
else:
    raw_vectors = panda_read(svd_path)
    print('Loaded SVD vectors')
vector_dictionary = {term[0]: term[1:] for term in raw_vectors.values[:, 1:]}


get_vectors = lambda theme: asarray([vector_dictionary[term] for term in dictionary[theme] if term in vector_dictionary])
p_vectors = get_vectors('positive_emotion')
n_vectors = get_vectors('negative_emotion')
vectors = vstack([p_vectors, n_vectors]).astype(float)
print('Generated vector sets')

centroids, distortion = kmeans(whiten(vectors), 2)

if len(centroids) < 1:
    raise ValueError('No centroids found')

distances = [
    [euclidean(centroid, vector) for vector in vectors] for centroid in centroids
]

colours = [1] * len(p_vectors) + [2] * len(n_vectors)
print('Computed K-means')

dist_1, dist_2 = distances
min_x, max_x = min(dist_1), max(dist_1)
min_y, max_y = min(dist_2), max(dist_2)

fig, ax = subplots()
img = ax.scatter(dist_1, dist_2, s=10, c=colours)
sep = ax.plot([min_x, max_x], [min_y, max_y], c='k')

x_lab = ax.set_xlabel('Distance from centroid 1')
y_lab = ax.set_ylabel('Distance from centroid 2')
title = ax.set_title('Distance from cluster centroids')

show()
