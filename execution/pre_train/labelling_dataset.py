from utilities.data_management import read_csv, move_to_root, make_path, load_execution_params, output_abusive_intent
from numpy import argsort, sum, any, where, zeros, all
from numpy.random import choice
from utilities.analysis import rescale_data
from model.analysis import compute_abusive_intent

move_to_root(4)
params = load_execution_params()

base = make_path('data/processed_data/') / params['dataset'] / 'analysis'
analysis_base = base / 'intent_abuse'
intent_base = base / 'intent'

english_mask = read_csv(intent_base / 'english_mask.csv', header=None)[0].values.astype(bool)
intent = read_csv(analysis_base / 'intent_predictions.csv', header=None)[0].values[english_mask]
abuse = read_csv(analysis_base / 'abuse_predictions.csv', header=None)[0].values[english_mask]

raw_contexts = read_csv(intent_base / 'contexts.csv').loc[english_mask]
contexts = raw_contexts['contexts'].values
print('Content loaded.')

print('intent', intent.shape, 'abuse', abuse.shape, 'contexts', contexts.shape, 'english mask', english_mask.shape)

# Rescale value range
intent = rescale_data(intent)
abuse = rescale_data(abuse)
print('Content prepared.')

# Remove wikipedia contexts used for training
non_wikipedia = raw_contexts['document_index'].values >= 0
contexts, intent, abuse = contexts[non_wikipedia], intent[non_wikipedia], abuse[non_wikipedia]

# Compute the euclidean norm of the (abuse, intent) vectors for each context
hybrid = compute_abusive_intent(intent, abuse)
hybrid_indexes = argsort(hybrid)
print('Finished computations.')

limit = .25
zone_width = .025
total_samples = 5000
num_zones = int(limit * 2 / zone_width)

abusive_intent = hybrid[hybrid_indexes]
# [strong_indexes] = where(any([abusive_intent <= limit, abusive_intent >= (1 - limit)], axis=0))
samples_per_zone = int(total_samples / num_zones)

zone_indexes = zeros((num_zones, samples_per_zone), dtype=int)
zone_indexes[:] = -1
for zone_number in range(num_zones):
    base = zone_number * zone_width + ((1 - limit * 2) if zone_number >= num_zones / 2 else 0)
    top = base + zone_width
    [indexes] = where(all([abusive_intent > base, abusive_intent <= top], axis=0))

    print(base, top, 'zone size', indexes.shape[0])
    chosen = choice(indexes.shape[0], min(samples_per_zone, indexes.shape[0]), replace=False)

    sample = indexes[chosen]
    zone_indexes[zone_number, :sample.shape[0]] = sample

print(zone_indexes)

print('samples per zone', samples_per_zone)
