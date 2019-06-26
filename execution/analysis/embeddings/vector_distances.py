from dask.dataframe import read_csv
from dask import delayed
from csv import QUOTE_NONE
from utilities.data_management import load_execution_params, make_path, move_to_root, check_existence
from scipy.spatial.distance import cosine, euclidean

# Define paths
move_to_root(4)
embed_name = load_execution_params()['fast_text_model']
embed_path = make_path('data/lexicons/fast_text/') / (embed_name + '.vec')
check_existence(embed_path)

# Define parameters
target_word = 'bitch'
max_cos_dist = .7

# Import data
embeddings = read_csv(embed_path, quoting=QUOTE_NONE, delimiter=' ', skiprows=1, header=None)
embeddings = embeddings.iloc[:, :-1]    # Ignore extra column

# Define slices
words = embeddings[[0]].rename(columns={0: 'words'})
vectors = embeddings.iloc[:, 1:]

# Normalize
mean = vectors.mean(axis=0)
std = vectors.std(axis=0)
vectors = (vectors - mean) / std

target = vectors[words['words'] == target_word].compute()

# Calculate cosine distances
calc_cosine = lambda vector: cosine(vector, target)
words['cosine_distances'] = vectors.map_partitions(lambda df: df.apply(calc_cosine, axis=1), meta=float)
vectors = vectors[words['cosine_distances'] < max_cos_dist]

# Calculate euclidean distances
calc_dist = lambda vector: euclidean(vector, target)
words['euclidean_distances'] = vectors.map_partitions(lambda df: df.apply(calc_dist, axis=1), meta=float)

close = words.nsmallest(n=50, columns='euclidean_distances').compute()
print(close)
