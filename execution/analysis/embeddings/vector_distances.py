from dask.dataframe import read_csv
from csv import QUOTE_NONE
from utilities.data_management import load_execution_params, make_path, move_to_root, check_existence
from matplotlib.pyplot import show, subplots
from utilities.analysis import get_nearest_neighbours

# Define paths
move_to_root(4)
embed_name = load_execution_params()['fast_text_model']
embed_path = make_path('data/lexicons/fast_text/') / (embed_name + '.vec')
check_existence(embed_path)

# Define parameters
target_word = 'dumbass'
max_cos_dist = 1

# Import data
embeddings = read_csv(embed_path, quoting=QUOTE_NONE, delimiter=' ', skiprows=1, header=None)
embeddings = embeddings.iloc[:, :-1]  # Ignore extra column

words, norm = get_nearest_neighbours(embeddings, target_word, silent=False)
print('Nearest Neighbours\n', words)
print(target_word, 'norm:', norm)

print(words.columns, type(words))
metrics = ['euclidean_distances', 'cosine_distances']
[axes] = words.hist(column=metrics, bins=40)

print(axes)

for ax, metric in zip(axes, metrics):
    ax.set_xlabel(metric.replace('_', ' '))
    ax.set_ylabel('Number of vectors')
    ax.set_title('Histogram of distances from ' + target_word)

show()
