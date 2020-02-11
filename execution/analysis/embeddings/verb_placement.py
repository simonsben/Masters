from utilities.data_management import move_to_root, make_path, check_existence, load_execution_params, open_w_pandas, \
    vector_to_file
from model.analysis import intent_verb_filename, cluster_verbs
from utilities.analysis import normalize_embeddings
from utilities.plotting import scatter_plot, show, scatter_3_plot, plot_dendrogram
from sklearn.decomposition import PCA
from empath import Empath
from numpy import asarray

move_to_root(4)

params = load_execution_params()
dataset = params['dataset']
model_name = params['fast_text_model']

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

action_tokens = action[:, 0]
action_vectors = normalize_embeddings(action[:, 1:].astype(float))
vector_to_file(action_tokens, verb_path_generator('action'))

categories = ['kill', 'leisure', 'exercise', 'communication']
thing = Empath()
is_polarizing = asarray([
    sum(thing.analyze(verb, categories=categories).values())
    for verb in action_tokens
]) != 0

desire_tokens = desire[:, 0]
desire_vectors = normalize_embeddings(desire[:, 1:].astype(float))
vector_to_file(desire_tokens, verb_path_generator('desire'))

pca = PCA(random_state=420)
reduced_action = pca.fit_transform(action_vectors)[is_polarizing]
reduced_desire = pca.fit_transform(desire_vectors)
print('Normalized verbs.')

num_verbs = 50
num_dimensions = 100

action_model = cluster_verbs(reduced_action, num_top_verbs=num_verbs, num_dimensions=num_dimensions)
desire_model = cluster_verbs(reduced_desire, num_top_verbs=num_verbs, num_dimensions=num_dimensions)
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
