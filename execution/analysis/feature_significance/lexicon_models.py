from utilities.data_management import move_to_root, open_w_pandas, make_path, check_existence, check_writable, \
    get_path_maps, load_xgboost_model, match_feature_weights
from utilities.analysis import get_feature_values
from matplotlib.pyplot import show
from utilities.plotting import feature_significance
from os import mkdir

move_to_root(4)

layer_names = ['hurtlex', 'subjectivity']

# Define file paths
dataset_name = '24k-abusive-tweets'
model_dir = make_path('data/models/') / dataset_name
figure_dir = make_path('figures/') / dataset_name / 'lexicon' / 'feature_significance/'
lexicon_dir = make_path('data/prepared_lexicon/')

# Check for files
if not figure_dir.exists():
    mkdir(figure_dir)

# Import data
maps = get_path_maps()
dir_maps = maps['layer_class']
name_maps = maps['layer_names']

# Get model and feature values
for layer in layer_names:
    model_path = model_dir / dir_maps[layer] / (layer + '.bin')
    lexicon_path = lexicon_dir / maps['lexicon_names'][layer]

    check_existence(model_path)
    check_existence(lexicon_path)

    features = open_w_pandas(lexicon_path, index_col=0)['word'].values

    model = load_xgboost_model(model_path)
    weights = get_feature_values(model)
    gains = get_feature_values(model, 'gain')

    feature_weights = match_feature_weights(features, weights)
    feature_gains = match_feature_weights(features, gains)

    # Stacked model
    feature_significance(feature_weights, name_maps[layer] + ' Weights',
                         filename=figure_dir / (layer + '_weight.png'))
    feature_significance(feature_gains, name_maps[layer] + ' Gains', is_weight=False, x_log=True,
                         filename=figure_dir / (layer + '_gain.png'))

show()
