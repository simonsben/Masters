from utilities.data_management import move_to_root, open_w_pandas, make_path, check_existence, check_writable, \
    load_execution_params
from model.training import train_xg_boost

move_to_root()

# Define file paths
dataset_name = load_execution_params()['dataset']
dataset_path = make_path('data/prepared_data/') / (dataset_name + '.csv')
predictions_dir = make_path('data/predictions') / dataset_name
train_path, test_path = predictions_dir / 'train.csv', predictions_dir / 'test.csv'
model_path = make_path('data/models/stacked/') / (dataset_name + '.bin')

# Check for files
check_existence(dataset_path)
check_existence(train_path)
check_existence(test_path)
check_writable(model_path)

# Import data
train_set = open_w_pandas(train_path)
test_set = open_w_pandas(test_path)

if 'stacked' in train_set:
    train_set.drop(columns='stacked', inplace=True)
    test_set.drop(columns='stacked', inplace=True)
train_set, test_set = train_set.values, test_set.values

labels = open_w_pandas(dataset_path)['is_abusive'].values.astype(bool)
train_labels, test_labels = labels[:len(train_set)], labels[len(train_set):]
print('Data loaded')

# Train predictor
classifier = train_xg_boost(train_set, train_labels, prepared=True, verb=1)
print('Model trained')

# Save model
classifier.save_model(str(model_path))
print('Model saved')
