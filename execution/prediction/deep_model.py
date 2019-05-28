from utilities.data_management import open_w_pandas, make_path, move_to_root, check_existence, split_sets, \
    to_numpy_array
from keras.models import load_model
from pandas import read_pickle

move_to_root()

# Define dataset paths
dataset = '24k-abusive-tweets'
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
train_predictions = open_w_pandas(train_pred_path)
test_predictions = open_w_pandas(test_pred_path)

# Open and split processed data
processed_data = read_pickle(data_path)
train_set, test_set = split_sets(processed_data, lambda docs: docs)
train_set, test_set = to_numpy_array([train_set, test_set])

# Load model and make predictions
model = load_model(str(model_path))
train_predictions['fast_text'] = model.predict(train_set)
test_predictions['fast_text'] = model.predict(test_set)
print('Data and model loaded, making predictions')

# Save predictions
train_predictions.to_csv(train_pred_path)
test_predictions.to_csv(test_pred_path)

# Print data summary
print(train_predictions.describe())
print(test_predictions.describe())
