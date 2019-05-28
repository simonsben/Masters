from utilities.data_management import move_to_root, open_w_pandas, make_path, check_existence, split_sets
from model.training import train_xg_boost

move_to_root()

dataset_name = '24k-abusive-tweets'
dataset_path = make_path('data/prepared_data/') / (dataset_name + '.csv')
predictions_dir = make_path('data/predictions') / dataset_name
train_path, test_path = predictions_dir / 'train.csv', predictions_dir / 'test.csv'

check_existence(dataset_path)
check_existence(train_path)
check_existence(test_path)

train_set = open_w_pandas(train_path).to_numpy()
test_set = open_w_pandas(test_path).to_numpy()
labels = open_w_pandas(dataset_path)['is_abusive'].to_numpy().astype(bool)
train_labels, test_labels = labels[:len(train_set)], labels[len(test_set):]
print('Data loaded')

classifier = train_xg_boost(train_set, train_labels, prepared=True, quiet=False)
print('Model trained')
