from utilities.data_management import read_csv, make_path, output_abusive_intent, load_vector, get_prediction_path
from numpy import argsort, where, all
from numpy.random import choice
from model.analysis import compute_abusive_intent
from config import dataset

context_path = make_path('data/processed_data/') / dataset / 'analysis' / 'intent' / 'contexts.csv'

abuse = load_vector(get_prediction_path('abuse'))
intent = load_vector(get_prediction_path('intent'))
abusive_intent = load_vector(get_prediction_path('abusive_intent'))
contexts = read_csv(context_path)['contexts'].values
print('Content loaded.')

# Rescale value range
print('Content prepared.')

# Compute the euclidean norm of the (abuse, intent) vectors for each context
hybrid_indexes = argsort(abusive_intent)
print('Finished computations.')

predictions = (abuse, intent, abusive_intent)

# Print records
num_records = 50
zone_size = .1

if (1 / zone_size) % 1 != 0:
    raise ValueError('Zone size must evenly divide 1')

print('Printing out samples from zones')
num_zones = int(1 / zone_size)
for index in range(num_zones):
    base = index * zone_size
    [indexes_in_range] = where(all([abusive_intent >= base, abusive_intent <= (base + zone_size)], axis=0))

    index_selection = indexes_in_range[
        choice(indexes_in_range.shape[0], min(num_records, indexes_in_range.shape[0]), replace=False)
    ]

    print('\nZone from', base, 'to', base + zone_size)
    output_abusive_intent(index_selection, predictions, contexts)
