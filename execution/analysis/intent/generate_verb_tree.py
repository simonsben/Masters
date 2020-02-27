from utilities.data_management import move_to_root, make_path, load_execution_params, load_vector, open_w_pandas, \
    check_existence, split_embeddings
from model.analysis import intent_verb_filename, get_polarizing_mask
from model.expansion.verb_tree import build_verb_tree, get_branch_leaves, check_for_labels
from model.analysis.clustering import reduce_and_cluster
from utilities.plotting import plot_dendrogram, show

move_to_root()
params = load_execution_params()
model_name = params['fast_text_model']
dataset = params['dataset']

base = make_path('data/processed_data/') / dataset / 'analysis'
embedding_dir = base / 'embeddings'
intent_dir = base / 'intent'

action_path = embedding_dir / intent_verb_filename('action', model_name)
desire_path = embedding_dir / intent_verb_filename('desire', model_name)
verb_path_generator = lambda name: embedding_dir / (name + '_verbs.csv')
figure_dir = make_path('figures') / dataset / 'analysis'

check_existence(action_path)
check_existence(desire_path)
print('Completed config.')

action = open_w_pandas(action_path).values
desire = open_w_pandas(desire_path).values
print('Loaded data.')

initial_mask = load_vector(intent_dir / 'intent_mask.csv')
contexts = open_w_pandas(intent_dir / 'contexts.csv')['contexts'].values
print('Content loaded.')

action_tokens, action_vectors = split_embeddings(action)
desire_tokens, desire_vectors = split_embeddings(desire)

is_polarizing = get_polarizing_mask(action_tokens)

action_model, reduced_action = reduce_and_cluster(action_vectors, is_polarizing, num_verbs=50)
desire_model, reduced_desire = reduce_and_cluster(desire_vectors, num_verbs=50)

action_tokens = action_tokens[is_polarizing][:action_model.n_leaves_]
desire_tokens = desire_tokens[:desire_model.n_leaves_]

action_tree = build_verb_tree(action_model, action_tokens)
action_leaves = get_branch_leaves(action_tree, ['kill', 'fight', 'eliminate'])
print(action_leaves)

target_desire_verbs = ['wish', 'hope', '']

desire_tree = build_verb_tree(desire_model, desire_tokens)
target_desire_verbs = check_for_labels(desire_tokens, target_desire_verbs)
desire_leaves = get_branch_leaves(desire_tree, target_desire_verbs)
print(desire_leaves)

plot_dendrogram(action_model, action_tokens, 'Action dendrogram')
plot_dendrogram(desire_model, desire_tokens, 'Desire dendrogram')

show()
