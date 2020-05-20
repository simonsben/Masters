from utilities.data_management import make_path, check_existence, open_w_pandas, load_vector
from numpy import any, zeros, sum, where, all
from utilities.plotting import confusion_matrix, show, scatter_plot

# Define import paths
base_path = make_path('data/processed_data/data_labelling/analysis/')
label_path = base_path / 'intent_abuse' / 'labels.csv'
prediction_path = base_path / 'intent_abuse' / 'intent_predictions.csv'
context_path = base_path / 'intent' / 'contexts.csv'

check_existence([label_path, prediction_path, context_path])
print('Config complete.')

# Load data
raw_labels = open_w_pandas(label_path)
predictions = load_vector(prediction_path)

raw_contexts = open_w_pandas(context_path)
contexts = raw_contexts['contexts'].values[raw_contexts.index.values >= 0]
print('Data loaded.')

# Get slice of labels
# Column order: SKIP, FALSE, TRUE, COMPUTED
labels = raw_labels.values[:, 1:-1]
valid_label_mask = any(labels >= 3, axis=1)

num_labels = raw_labels.shape[0]
num_valid_labels = sum(valid_label_mask)

# Get labels as a boolean array
boolean_labels = raw_labels['1'].values >= 3
valid_labels = boolean_labels[valid_label_mask]

# Convert predictions to a boolean array
truncated_predictions = predictions[:num_labels]
boolean_predictions = truncated_predictions > .5
valid_predictions = boolean_predictions[valid_label_mask]

# Get array of correct predictions
prediction_is_correct = boolean_labels == boolean_predictions
valid_correct_predictions = prediction_is_correct[valid_label_mask]


print('\nAll labels:')
print('Correct', sum(prediction_is_correct), 'of', num_labels)
print('Accuracy', sum(prediction_is_correct) / num_labels)

print('\nValidated labels:')
print('Correct', sum(valid_correct_predictions), 'of', num_valid_labels)
print('Accuracy', sum(valid_correct_predictions) / num_valid_labels)

[false_negatives] = where(
    all([valid_labels == True, valid_predictions == False], axis=0)
)
[false_positives] = where(
    all([valid_labels == False, valid_predictions == True], axis=0)
)

valid_contexts = contexts[:num_labels][valid_label_mask]
bundled_labels = raw_labels.values[valid_label_mask]


print('\nLabel order: SKIP, FALSE, TRUE, COMPUTED')

print('\nFalse negatives:')
for index in false_negatives:
    print(index, truncated_predictions[valid_label_mask][index], bundled_labels[index], valid_contexts[index])

print('\nFalse positives:')
for index in false_positives:
    print(index, truncated_predictions[valid_label_mask][index], bundled_labels[index], valid_contexts[index])


# Plot predictions vs. labels
ax_titles = ('Predictions', 'Effective labels')

confusion_matrix(boolean_predictions[valid_label_mask], boolean_labels[valid_label_mask], 'Intent prediction validation')
scatter_plot((truncated_predictions[valid_label_mask], raw_labels['rating'].values[valid_label_mask]),
             'Intent prediction validation', ax_titles=ax_titles)
show()
