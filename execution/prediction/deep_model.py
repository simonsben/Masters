from utilities.data_management import open_w_pandas, make_path, move_to_root, check_existence, split_sets, \
    to_numpy_array, load_execution_params
from keras.models import load_model
from pandas import read_pickle
from numpy import where

move_to_root()

# Define dataset paths
dataset = load_execution_params()['dataset']
pred_dir = make_path('data/predictions') / dataset
data_path = make_path('data/processed_data/') / dataset / 'derived/fast_text.pkl'
model_path = make_path('data/models/') / dataset / 'derived/fast_text.h5'
train_pred_path, test_pred_path = pred_dir / 'train.csv', pred_dir / 'test.csv'

# Check existence of files
check_existence(train_pred_path)
check_existence(test_pred_path)
check_existence(model_path)
check_existence(data_path)

# Open predictions
train_predictions = open_w_pandas(train_pred_path, index_col=0)
test_predictions = open_w_pandas(test_pred_path, index_col=0)

# Open and split processed data
processed_data = read_pickle(data_path)
train_set, test_set = split_sets(processed_data, lambda docs: docs)
train_set, test_set = to_numpy_array([train_set, test_set])
print('Data loaded')

# Load model and make predictions
model = load_model(str(model_path))
train_predictions['fast_text'] = where(model.predict(train_set) > .5, 1, 0)
test_predictions['fast_text'] = where(model.predict(test_set) > .5, 1, 0)

# Save predictions
train_predictions.to_csv(train_pred_path)
test_predictions.to_csv(test_pred_path)

# Print data summary
print(train_predictions.describe())
print(test_predictions.describe())
