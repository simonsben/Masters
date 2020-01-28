from utilities.data_management import read_csv, move_to_root, make_path, load_execution_params
from numpy import asarray, argsort, savetxt
from scipy.linalg import norm
from utilities.analysis import rescale_data

move_to_root(4)
params = load_execution_params()
is_subset = False

base = make_path('data/processed_data/') / params['dataset'] / 'analysis'
analysis_base = base / 'intent_abuse'

intent = read_csv(analysis_base / 'intent_predictions.csv', header=None)[0].values
abuse = read_csv(analysis_base / 'abuse_predictions.csv', header=None)[0].values
contexts = read_csv(base / 'intent' / 'contexts.csv')['contexts'].values
print('Content loaded.')

# Remove small documents
if is_subset:
    max_docs = int(contexts.shape[0] * .25)
    subset_mask = asarray([(len(context.split(' ')) > 4) for context in contexts]).astype(bool)
    subset_mask[max_docs:] = False

    abuse = abuse[subset_mask]
    contexts = contexts[subset_mask]
print('Filtered vector shapes', abuse.shape, intent.shape, contexts.shape)

# Rescale value range
intent = rescale_data(intent)
abuse[abuse > 1] = 1
print('Content prepared.')

# Compute the euclidean norm of the (abuse, intent) vectors for each context
hybrid = asarray([
    norm((_intent, _abuse)) for _intent, _abuse in zip(intent, abuse)
])
hybrid_indexes = argsort(hybrid)
print('Finished computations.')


def print_out(index_set, filename):
    index_set = asarray(list(index_set))
    savetxt(analysis_base / (filename + '.csv'), index_set, delimiter=',', fmt='%d')

    print('%10s %8s %8s %8s  %s' % ('index', 'hybrid', 'intent', 'abuse', 'context'))
    for index in index_set:
        print('%10d %8.4f %8.4f %8.4f  %s' % (index, hybrid[index], intent[index], abuse[index], contexts[index]))


# Print records
num_records = 50

print('\nHigh')
print_out(reversed(hybrid_indexes[-num_records:]), 'high_indexes')

print('\nLow')
print_out(hybrid_indexes[:num_records], 'low_indexes')
