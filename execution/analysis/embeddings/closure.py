from dask.dataframe import read_csv
from utilities.data_management import load_execution_params, make_path, move_to_root, check_existence, make_dir
from utilities.analysis import get_nearest_neighbours, svd_embeddings
from model.analysis import cluster_neighbours
from numpy import savetxt


# Define parameters
seed_terms = ['bitch']
# seed_terms = ['fucking', 'bitch', 'fuck', 'bitches', 'ass', 'fucked',
# 'shit', 'stupid', 'pussy', 'hoes', 'idiot', 'hoe']
closure_set = set(seed_terms)
to_be_closed = seed_terms.copy()

# Define paths
move_to_root(4)
params = load_execution_params()
embed_name = params['fast_text_model']
data_name = params['dataset']
embed_path = make_path('data/prepared_lexicon/') / (embed_name + '.csv')
dest_dir = make_path('data/processed_data/') / data_name / 'analysis' / 'closure'

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

    terms, target = get_nearest_neighbours(embeddings, round_target, n_words=250)
    # print(terms)

    close_terms = cluster_neighbours(terms, True)

    new_terms = [term for term in close_terms if term not in closure_set]
    closure_set.update(new_terms)
    to_be_closed = to_be_closed + new_terms
    print('Adding', new_terms)

print('Set closed', closure_set)

data = [seed_terms, list(closure_set)]
savetxt((dest_dir / 'closure.csv'), data)
