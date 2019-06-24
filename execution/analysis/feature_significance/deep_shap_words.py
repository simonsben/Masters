from utilities.data_management import move_to_root, make_path, load_execution_params, open_w_pandas, load_dataset_params
from os import mkdir
from matplotlib.pyplot import show
from model.training import load_attention
from fastText import load_model
from utilities.plotting import word_importance
from numpy import zeros

move_to_root(4)

# Define file paths
params = load_execution_params()
max_tokens = load_dataset_params()['max_document_tokens']
dataset_name = params['dataset']
ft_name = params['fast_text_model']

shap_dir = make_path('figures/') / dataset_name / 'derived' / 'shap' / 'shap_words'
data_path = make_path('data/prepared_data/') / (dataset_name + '.csv')
model_path = make_path('data/models/') / dataset_name / 'derived/' / 'fast_text.h5'
ft_path = make_path('data/lexicons/fast_text/') / (ft_name + '.bin')

# Check for files
if not shap_dir.exists(): mkdir(shap_dir)

# Get model and feature values
model = load_attention(model_path)
print('Deep model loaded')

ft_model = load_model(str(ft_path))
print('FastText model loaded')

dataset = open_w_pandas(data_path).sample(100)['document_content']
print('Dataset loaded, starting')

vectorized = zeros((len(dataset), max_tokens, ft_model.get_dimension()))
for ind, document in enumerate(dataset):
    for t_ind, token in enumerate(document.split(' ')):
        vectorized[ind, t_ind] = ft_model.get_word_vector(token)
# vectorized = [
#     [ft_model.get_word_vector(word) for word in document.split(' ')]
#     for document in dataset.values
# ]
print('Sample vectorized')

path_gen = lambda ind: shap_dir / ('word_shap_' + str(ind) + '.png')
word_importance(dataset.values, vectorized, model=model, path_gen=path_gen)

show()
