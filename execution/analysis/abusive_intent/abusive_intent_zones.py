from utilities.data_management import read_csv, make_path, output_abusive_intent
from numpy import argsort, where, all
from numpy.random import choice
from utilities.analysis import rescale_data
from model.analysis import compute_abusive_intent
import config

base = make_path('data/processed_data/') / config.dataset / 'analysis'
analysis_base = base / 'intent_abuse'

intent = read_csv(analysis_base / 'intent_predictions.csv', header=None)[0].values
abuse = read_csv(analysis_base / 'abuse_predictions.csv', header=None)[0].values
contexts = read_csv(base / 'intent' / 'contexts.csv')['contexts'].values
print('Content loaded.')

# Rescale value range
print('Content prepared.')

# Compute the euclidean norm of the (abuse, intent) vectors for each context
hybrid = compute_abusive_intent(intent, abuse)
hybrid = rescale_data(hybrid)
hybrid_indexes = argsort(hybrid)
print('Finished computations.')

predictions = (abuse, intent, hybrid)

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
        choice(indexes_in_range.shape[0], min(num_records, indexes_in_range.shape[0]), replace=False)
    ]

    print('\nZone from', base, 'to', base + zone_size)
    output_abusive_intent(index_selection, predictions, contexts)
