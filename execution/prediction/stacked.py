from utilities.data_management import move_to_root, open_w_pandas, make_path, check_existence, check_writable
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

move_to_root()

# Define file paths
dataset_name = '24k-abusive-tweets'
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
train_set = open_w_pandas(train_path, index_col=0)
test_set = open_w_pandas(test_path, index_col=0)
labels = open_w_pandas(dataset_path)['is_abusive'].to_numpy().astype(bool)
train_labels, test_labels = labels[:len(train_set)], labels[len(test_set):]
print('Data loaded, making predictions')

# Load model
model = XGBClassifier(objective='binary:logistic', n_estimators=600, silent=True)
model.load_model(str(model_path))
model._le = LabelEncoder().fit([0, 1])

# Make predictions
train_set['stacked'] = model.predict(train_set.to_numpy())
test_set['stacked'] = model.predict(test_set.to_numpy())

# Save predictions
train_set.to_csv(train_path)
test_set.to_csv(test_path)
print('Predictions saved')

print(train_set.describe())
print(test_set.describe())
