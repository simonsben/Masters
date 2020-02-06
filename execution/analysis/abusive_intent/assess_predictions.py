from utilities.data_management import read_csv, move_to_root, make_path, load_execution_params, output_abusive_intent
from numpy import argsort
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
hybrid = compute_abusive_intent(intent, abuse, True)
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
