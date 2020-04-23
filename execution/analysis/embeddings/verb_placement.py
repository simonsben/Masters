from utilities.data_management import make_path, check_existence, open_w_pandas, vector_to_file, \
    split_embeddings
from model.analysis import intent_verb_filename, get_polarizing_mask
from model.analysis.clustering import reduce_and_cluster
from utilities.analysis import normalize_embeddings
from utilities.plotting import show, plot_dendrogram
import config

dataset = config.dataset
model_name = config.fast_text_model

base_dir = make_path('data/processed_data') / dataset / 'analysis' / 'embeddings'
action_path = base_dir / intent_verb_filename('action', model_name)
desire_path = base_dir / intent_verb_filename('desire', model_name)
verb_path_generator = lambda name: base_dir / (name + '_verbs.csv')
figure_dir = make_path('figures') / dataset / 'analysis'

check_existence(action_path)
check_existence(desire_path)
print('Completed config.')

action = open_w_pandas(action_path).values
desire = open_w_pandas(desire_path).values
print('Loaded data.')

action_tokens, action_vectors = split_embeddings(action)
vector_to_file(action_tokens, verb_path_generator('action'))

is_polarizing = get_polarizing_mask(action_tokens)

desire_tokens, desire_vectors = split_embeddings(desire)
vector_to_file(desire_tokens, verb_path_generator('desire'))

action_model, reduced_action = reduce_and_cluster(action_vectors, is_polarizing)
desire_model, reduced_desire = reduce_and_cluster(desire_vectors)
print('Completed clustering')

plot_dendrogram(
    action_model, action_tokens[is_polarizing][:action_model.n_leaves_], 'Dendrogram of action verbs',
    filename=(figure_dir / 'action_verb_clustering.png')
)
plot_dendrogram(
    desire_model, desire_tokens[:desire_model.n_leaves_], 'Dendrogram of desire verbs',
    filename=(figure_dir / 'desire_verb_clustering.png')
)

show()
