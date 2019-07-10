from utilities.data_management import move_to_root, open_w_pandas, make_path, check_existence, check_writable, \
    get_path_maps, load_execution_params
from utilities.plotting import confusion_matrix
from matplotlib.pyplot import show, close
from os import mkdir

move_to_root()

# Define file paths
dataset_name = load_execution_params()['dataset']
dataset_path = make_path('data/prepared_data/') / (dataset_name + '.csv')
predictions_dir = make_path('data/predictions') / dataset_name
threshold_path = predictions_dir / 'thresholds.csv'
train_path, test_path = predictions_dir / 'train.csv', predictions_dir / 'test.csv'
model_path = make_path('data/models/stacked/') / (dataset_name + '.bin')
figure_dir = make_path('figures/') / dataset_name

# Check for files
check_existence(dataset_path)
check_existence(train_path)
check_existence(test_path)
check_existence(threshold_path)
check_writable(figure_dir)

# Import data
train_set = open_w_pandas(train_path)
test_set = open_w_pandas(test_path)
labels = open_w_pandas(dataset_path)['is_abusive'].to_numpy().astype(bool)
train_labels, test_labels = labels[:len(train_set)], labels[len(train_set):]
print('Data loaded, generating figures')

maps = get_path_maps()
dir_maps = maps['layer_class']
name_maps = maps['layer_names']

layers = train_set.columns

for ind, layer in enumerate(layers):
    dir_map = dir_maps[layer] if layer in dir_maps else 'lexicon'
    base = figure_dir / dir_map / 'confusion_matrix'
    if not base.exists():
        mkdir(base)

    name_map = name_maps[layer] if layer in name_maps else str(layer).replace('_', ' ').capitalize()
    confusion_matrix(train_set[layer].values, train_labels,
                     name_map + ' Predictor, training data',
                     base / (layer + '_train.png'))
    confusion_matrix(test_set[layer].values, test_labels,
                     name_map + ' Predictor, test data',
                     base / (layer + '_test.png'))

# show()
close('all')
