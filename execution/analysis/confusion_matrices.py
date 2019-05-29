from utilities.data_management import move_to_root, open_w_pandas, make_path, check_existence, check_writable, get_path_maps
from utilities.plotting import confusion_matrix
from matplotlib.pyplot import show

# TODO check data and plotting, values seem wrong..

move_to_root()

# Define file paths
dataset_name = '24k-abusive-tweets'
dataset_path = make_path('data/prepared_data/') / (dataset_name + '.csv')
predictions_dir = make_path('data/predictions') / dataset_name
train_path, test_path = predictions_dir / 'train.csv', predictions_dir / 'test.csv'
model_path = make_path('data/models/stacked/') / (dataset_name + '.bin')
figure_dir = make_path('figures/') / dataset_name

# Check for files
check_existence(dataset_path)
check_existence(train_path)
check_existence(test_path)
check_writable(figure_dir)

# Import data
train_set = open_w_pandas(train_path, index_col=0)
test_set = open_w_pandas(test_path, index_col=0)
labels = open_w_pandas(dataset_path)['is_abusive'].to_numpy().astype(bool)
train_labels, test_labels = labels[:len(train_set)], labels[len(train_set):]
print('Data loaded, generating figures')

maps = get_path_maps()
dir_maps = maps['layer_class']
name_maps = maps['layer_names']

layers = train_set.columns

for layer in layers:
    base = figure_dir / dir_maps[layer]

    confusion_matrix(train_set[layer].to_numpy(), train_labels,
                     name_maps[layer] + ' Predictor, training data',
                     base / ('cm_' + layer + '_train.png'))
    confusion_matrix(train_set[layer].to_numpy(), train_labels,
                     name_maps[layer] + ' Predictor, test data',
                     base / ('cm_' + layer + '_test.png'))

# show()
