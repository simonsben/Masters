from utilities.data_management import read_csv, make_path, output_abusive_intent, load_vector
from model.analysis import estimate_joint_cumulative
from numpy import argsort
from config import dataset

base = make_path('data/processed_data/') / dataset / 'analysis'
analysis_base = base / 'intent_abuse'
intent_base = base / 'intent'
prediction_path = lambda target: analysis_base / (target + '_predictions.csv')

abuse = load_vector(prediction_path('abuse'))
intent = load_vector(prediction_path('intent'))
abusive_intent = load_vector(prediction_path('abusive_intent'))

# joint = estimate_joint_cumulative(abuse, intent)
# abusive_intent = joint(abuse, intent)

raw_contexts = read_csv(intent_base / 'contexts.csv')
contexts = raw_contexts['contexts'].values
print('Content loaded.')

print('intent', intent.shape, 'abuse', abuse.shape, 'contexts', contexts.shape, 'english mask')

indexes = argsort(abusive_intent)
predictions = (abuse, intent, abusive_intent)

# Remove wikipedia contexts used for training
# non_wikipedia = raw_contexts['document_index'].values >= 0
# contexts, intent, abuse = contexts[non_wikipedia], intent[non_wikipedia], abuse[non_wikipedia]

# Print records
num_records = 50

print('\nHigh')
output_abusive_intent(reversed(indexes[-num_records:]), predictions, contexts)

print('\nLow')
output_abusive_intent(indexes[:num_records], predictions, contexts)
