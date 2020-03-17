from utilities.data_management import read_csv, move_to_root, make_path, load_execution_params, output_abusive_intent
from numpy import argsort, sum
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
contexts = read_csv(intent_base / 'contexts.csv')['contexts'].values[english_mask]
print('Content loaded.')

print('intent', intent.shape, 'abuse', abuse.shape, 'contexts', contexts.shape, 'english mask', english_mask.shape)

# Rescale value range
intent = rescale_data(intent)
abuse = rescale_data(abuse)
print('Content prepared.')

# Compute the euclidean norm of the (abuse, intent) vectors for each context
hybrid = compute_abusive_intent(intent, abuse)
hybrid_indexes = argsort(hybrid)
print('Finished computations.')

gen_filename = lambda name: analysis_base / (name + '.csv')
predictions = (hybrid, intent, abuse)

# Print records
num_records = 100

print('\nHigh')
output_abusive_intent(reversed(hybrid_indexes[-num_records:]), predictions, contexts, gen_filename('high_indexes'))

print('\nLow')
output_abusive_intent(hybrid_indexes[:num_records], predictions, contexts, gen_filename('low_indexes'))
