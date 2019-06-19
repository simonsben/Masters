from shap import DeepExplainer
from utilities.data_management import move_to_root, make_path, load_execution_params, split_sets, open_w_pandas, \
    to_numpy_array
from utilities.plotting import feature_significance
from os import mkdir
from pandas import read_pickle
from matplotlib.pyplot import show
from model.training import load_attention

move_to_root(4)

# Define file paths
dataset_name = load_execution_params()['dataset']
figure_base = make_path('figures/') / dataset_name / 'derived'
shap_dir = figure_base / 'shap'
data_path = make_path('data/prepared_data/') / (dataset_name + '.csv')
vectorized_path = make_path('data/processed_data/') / dataset_name / 'derived/' / 'fast_text.pkl'
model_path = make_path('data/models/') / dataset_name / 'derived/' / 'fast_text.h5'

# Check for files
if not shap_dir.exists(): mkdir(shap_dir)

# Get model and feature values
model = load_attention(model_path)
print('Model loaded')

dataset = open_w_pandas(data_path)
dataset['vectorized_content'] = read_pickle(vectorized_path)

(train, test) = split_sets(dataset['vectorized_content'])
[train_sample, test_sample] = to_numpy_array([train.sample(500), test.sample(100)])
print('Loaded dataset')

explainer = DeepExplainer(model, train_sample)
print('Explainer initialized')

[shap_values] = explainer.shap_values(test_sample)
shap_values = shap_values.mean(axis=0).mean(axis=0)

features = [('Dimension ' + str(i)) for i in range(len(shap_values))]
feature_weights = sorted(
    [(feature, weight) for feature, weight in zip(features, shap_values)],
    reverse=True, key=lambda doc: doc[1]
)

# Deep model
feature_significance(feature_weights, 'FastText BiLSTM SHAP Weights',
                     filename=shap_dir / 'deep_weight.png')

# show()
