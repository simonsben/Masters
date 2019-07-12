from dask.dataframe import read_csv
from utilities.data_management import load_execution_params, make_path, move_to_root, check_existence, make_dir
# from matplotlib.pyplot import show, savefig, tight_layout
from utilities.analysis import get_nearest_neighbours, svd_embeddings
from model.analysis import cluster_neighbours
# from utilities.plotting import scatter_plot, plot_embedding_rep, scatter_3_plot


# Define parameters
target_word = 'bitch'
closure_set = {target_word}
to_be_closed = [target_word]

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

while len(to_be_closed) > 0:
    round_target = to_be_closed.pop(0)
    print('Closing', round_target)

    terms, target = get_nearest_neighbours(embeddings, target_word, n_words=250)
    print(terms)

    close_terms = cluster_neighbours(terms, True)

    new_terms = [term for term in close_terms if term not in closure_set]
    closure_set.update(new_terms)
    to_be_closed = to_be_closed + new_terms
    print('Adding', new_terms)

print('Set closed', closure_set)
