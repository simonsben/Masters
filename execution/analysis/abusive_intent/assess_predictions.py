from utilities.data_management import read_csv, make_path, output_abusive_intent, load_vector
from numpy import argsort, sum
from model.analysis import compute_abusive_intent
import config

base = make_path('data/processed_data/') / config.dataset / 'analysis'
analysis_base = base / 'intent_abuse'
intent_base = base / 'intent'
prediction_path = lambda target: analysis_base / (target + '_predictions.csv')



# english_mask = load_vector(intent_base / 'english_mask.csv')
abuse = load_vector(prediction_path('abuse'))
intent = load_vector(prediction_path('intent'))
abusive_intent = load_vector(prediction_path('abusive_intent'))

raw_contexts = read_csv(intent_base / 'contexts.csv')
contexts = raw_contexts['contexts'].values
print('Content loaded.')

print('intent', intent.shape, 'abuse', abuse.shape, 'contexts', contexts.shape, 'english mask')

indexes = argsort(abusive_intent)
predictions = (abuse, intent, abusive_intent)

# Remove wikipedia contexts used for training
# non_wikipedia = raw_contexts['document_index'].values >= 0
# contexts, intent, abuse = contexts[non_wikipedia], intent[non_wikipedia], abuse[non_wikipedia]

# Compute the euclidean norm of the (abuse, intent) vectors for each context
# hybrid = compute_abusive_intent(intent, abuse)
# hybrid_indexes = argsort(hybrid)
# print('Finished computations.')
#
# gen_filename = lambda name: analysis_base / (name + '.csv')
# predictions = (hybrid, intent, abuse)

# Print records
num_records = 50

print('\nHigh')
output_abusive_intent(reversed(indexes[-num_records:]), predictions, contexts)

print('\nLow')
output_abusive_intent(indexes[:num_records], predictions, contexts)
