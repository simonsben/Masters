from dask.dataframe import read_csv
from utilities.data_management import load_execution_params, make_path, move_to_root, check_existence, make_dir, \
    prepare_csv_writer
from utilities.analysis import get_nearest_neighbours, svd_embeddings
from model.analysis import cluster_neighbours


# Define parameters
seed_terms = ['bitch']
# seed_terms = ['fucking', 'bitch', 'fuck', 'bitches', 'ass',
# 'fucked', 'shit', 'stupid', 'pussy', 'hoes', 'idiot', 'hoe']
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
print('Embeddings ready, running closure')

added_terms = []
root_target = None

while len(to_be_closed) > 0:
    target_word = to_be_closed.pop(0)
    print('Closing', target_word)

    terms, target = get_nearest_neighbours(embeddings, target_word, n_words=250)

    if len(terms) < 2:
        continue

    if root_target is None:
        root_target = target

    close_terms, distances = cluster_neighbours(terms, True, root_target)

    new_terms = [term for term in close_terms if term not in closure_set]
    closure_set.update(new_terms)
    to_be_closed = to_be_closed + new_terms
    print('Adding', new_terms)
    print(distances)

print('Set closed', closure_set)

# Save data
writer, fl = prepare_csv_writer(dest_dir / 'closure_set_' + str(len(seed_terms)) + '.csv')
writer.writerow(seed_terms)
for w_expanded in added_terms:
    writer.writerow(w_expanded)

fl.close()
