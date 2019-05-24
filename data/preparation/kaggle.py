from pandas import read_csv, concat
from utilities.data_management import make_path
from utilities.pre_processing import remove_unicode_values

# Define filenames
base_directory = make_path('../datasets/kaggle/')
test_dataset = read_csv(base_directory / 'test.csv')
train_dataset = read_csv(base_directory / 'train.csv')
test_labels = read_csv(base_directory / 'test_labels.csv')

# Add labels to test dataset
test_cols = test_labels.columns[1:]
for col in test_cols:
    test_dataset[col] = test_labels[col]
test_labels = None  # Make memory available for trash collection

# Combine training and testing datasets
dataset = concat([train_dataset, test_dataset])
dataset['comment_text'] = remove_unicode_values(dataset['comment_text'])

# Save all data
dataset.to_csv(base_directory / 'kaggle.csv')
