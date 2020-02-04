from utilities.data_management import read_csv, move_to_root, make_path, load_execution_params, output_abusive_intent
from numpy import argsort, where, all
from numpy.random import choice
from utilities.analysis import rescale_data
from model.analysis import compute_abusive_intent

move_to_root(4)
params = load_execution_params()

base = make_path('data/processed_data/') / params['dataset'] / 'analysis'
analysis_base = base / 'intent_abuse'

intent = read_csv(analysis_base / 'intent_predictions.csv', header=None)[0].values
abuse = read_csv(analysis_base / 'abuse_predictions.csv', header=None)[0].values
contexts = read_csv(base / 'intent' / 'contexts.csv')['contexts'].values
print('Content loaded.')

# Rescale value range
intent = rescale_data(intent)
abuse = rescale_data(abuse)
print('Content prepared.')

# Compute the euclidean norm of the (abuse, intent) vectors for each context
hybrid = compute_abusive_intent(intent, abuse, False)
hybrid = rescale_data(hybrid)
hybrid_indexes = argsort(hybrid)
print('Finished computations.')

# gen_filename = lambda name: analysis_base / (name + '.csv')
predictions = (hybrid, intent, abuse)

# Print records
num_records = 50
zone_size = .1

if (1 / zone_size) % 1 != 0:
    raise ValueError('Zone size must evenly divide 1')

print('Printing out samples from zones')
num_zones = int(1 / zone_size)
for index in range(num_zones):
    base = index * zone_size
    [indexes_in_range] = where(all([hybrid >= base, hybrid <= (base + zone_size)], axis=0))

    index_selection = indexes_in_range[
        choice(indexes_in_range.shape[0], num_records, replace=False)
    ]

    print('\nZone from', base, 'to', base + zone_size)
    output_abusive_intent(index_selection, predictions, contexts)
