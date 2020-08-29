from utilities.data_management import read_csv, make_path, output_abusive_intent, load_vector
from model.analysis import estimate_joint_cumulative, get_norm
from utilities.plotting import hist_plot, show, set_font_size
from numpy import argsort, logical_not
from config import dataset

latex_output = True
force_english = False
force_non_english = False
norm_type = 'product'

base = make_path('data/processed_data/') / dataset / 'analysis'
analysis_base = base / 'intent_abuse'
intent_base = base / 'intent'
prediction_path = lambda target: analysis_base / (target + '_predictions.csv')
figure_path = make_path('figures') / dataset / 'analysis' / ('abusive_intent_%s_norm.png' % norm_type)
is_english_path = intent_base / 'english_mask.csv'

is_english = load_vector(is_english_path).astype(bool)
abuse = load_vector(prediction_path('abuse'))
intent = load_vector(prediction_path('intent'))
# abusive_intent = load_vector(prediction_path('abusive_intent'))

# joint = estimate_joint_cumulative(abuse, intent)
# abusive_intent = joint(abuse, intent)

norm = get_norm(norm_type)
abusive_intent = norm((abuse, intent), axis=0)

raw_contexts = read_csv(intent_base / 'contexts.csv')
contexts = raw_contexts['contexts'].values
document_indexes = raw_contexts['document_index'].values
print('Content loaded.')

print('intent', intent.shape, 'abuse', abuse.shape, 'contexts', contexts.shape)

if force_english:
    indexes = argsort(abusive_intent[is_english])
    predictions = (abuse[is_english], intent[is_english], abusive_intent[is_english])
    contexts = contexts[is_english]
elif force_non_english:
    not_english = logical_not(is_english)
    indexes = argsort(abusive_intent[not_english])
    predictions = (abuse[not_english], intent[not_english], abusive_intent[not_english])
    contexts = contexts[not_english]
else:
    mask = document_indexes >= 0

    indexes = argsort(abusive_intent[mask])
    predictions = (abuse[mask], intent[mask], abusive_intent[mask])

# Remove wikipedia contexts used for training
# non_wikipedia = raw_contexts['document_index'].values >= 0
# contexts, intent, abuse = contexts[non_wikipedia], intent[non_wikipedia], abuse[non_wikipedia]


# Print records
num_records = 25

print('\nHigh')
output_abusive_intent(reversed(indexes[-num_records:]), predictions, contexts, latex_style=latex_output)

print('\nLow')
output_abusive_intent(indexes[:num_records], predictions, contexts, latex_style=latex_output)

set_font_size(16)
hist_plot(abusive_intent, ('Abusive intent histogram with %s norm' % norm_type), figure_path)

show()
