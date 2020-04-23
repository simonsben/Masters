from utilities.data_management import read_csv, move_to_root, make_path, output_abusive_intent
from numpy import argsort, sum
from utilities.analysis import rescale_data
from model.analysis import compute_abusive_intent
import config

move_to_root(4)

base = make_path('data/processed_data/') / config.dataset / 'analysis'
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

gen_filename = lambda name: analysis_base / (name + '.csv')
predictions = (hybrid, intent, abuse)

# Print records
num_records = 50

print('\nHigh')
output_abusive_intent(reversed(hybrid_indexes[-num_records:]), predictions, contexts, gen_filename('high_indexes'))

print('\nLow')
output_abusive_intent(hybrid_indexes[:num_records], predictions, contexts, gen_filename('low_indexes'))
