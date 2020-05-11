from utilities.data_management import make_path, check_existence, open_w_pandas, load_vector
from numpy import any, zeros, sum, where, all
from utilities.plotting import confusion_matrix, show, scatter_plot

base_path = make_path('data/processed_data/data_labelling/analysis/')
label_path = base_path / 'intent_abuse' / 'labels.csv'
prediction_path = base_path / 'intent_abuse' / 'intent_predictions.csv'
context_path = base_path / 'intent' / 'contexts.csv'

check_existence([label_path, prediction_path, context_path])
print('Config complete.')

# Load data
raw_labels = open_w_pandas(label_path)
raw_contexts = open_w_pandas(context_path)
contexts = raw_contexts['contexts'].values[raw_contexts.index.values >= 0]
predictions = load_vector(prediction_path)
print('Data loaded.')


labels = raw_labels.values[:, 1:-1]
valid_label_mask = any(labels >= 3, axis=1)
num_labels = raw_labels.shape[0]
num_valid_labels = sum(valid_label_mask)

clean_labels = zeros(num_labels, dtype=bool)
clean_labels[raw_labels['1'].values >= 3] = True
valid_labels = clean_labels[valid_label_mask]

related_predictions = predictions[:num_labels]
rounded_predictions = related_predictions > .6
valid_predictions = rounded_predictions[valid_label_mask]

correct_predictions = rounded_predictions == clean_labels
valid_correct_predictions = correct_predictions[valid_label_mask]

print('\nAll labels:')
print('Correct', sum(correct_predictions), 'of', num_labels)
print('Accuracy', sum(correct_predictions) / num_labels)

print('\nValidated numbers:')
print('Correct', sum(valid_correct_predictions), 'of', num_valid_labels)
print('Accuracy', sum(valid_correct_predictions) / num_valid_labels)

[false_negatives] = where(
    all([valid_labels == True, valid_predictions == False], axis=0)
)
[false_positives] = where(
    all([valid_labels == False, valid_predictions == True], axis=0)
)

valid_contexts = contexts[:num_labels][valid_label_mask]
output_labels = raw_labels.values[valid_label_mask]

print('\nLabel order: SKIP, FALSE, TRUE, COMPUTED')

print('\nFalse negatives:')
for index in false_negatives:
    print(index, related_predictions[valid_label_mask][index], output_labels[index],  valid_contexts[index])


print('\nFalse positives:')
for index in false_positives:
    print(index, related_predictions[valid_label_mask][index], output_labels[index],  valid_contexts[index])


confusion_matrix(rounded_predictions[valid_label_mask], clean_labels[valid_label_mask], 'Intent prediction validation')
scatter_plot((related_predictions[valid_label_mask], raw_labels['rating'].values[valid_label_mask]), 'Intent prediction validation')
show()
