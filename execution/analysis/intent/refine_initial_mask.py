from utilities.data_management import move_to_root, make_path, load_execution_params, load_vector, open_w_pandas, \
    check_existence, split_embeddings, read_csv
from model.analysis import intent_verb_filename
from model.expansion.verb_tree import build_verb_tree, get_branch_leaves, check_for_labels
from model.analysis.clustering import reduce_and_cluster
from numpy import asarray, logical_not, all, sum
from utilities.plotting import plot_dendrogram, show

target_action_verbs = ['kill', 'fight']
target_desire_verbs = ['want', 'going', 'have', 'must']

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
intent_frames = read_csv(intent_dir / 'intent_frame.csv', header=None).values
print('Content loaded.')

action_tokens, action_vectors = split_embeddings(action)
desire_tokens, desire_vectors = split_embeddings(desire)

action_model, reduced_action = reduce_and_cluster(action_vectors, num_verbs=None)
desire_model, reduced_desire = reduce_and_cluster(desire_vectors, num_verbs=None)
print('Verbs clustered')

action_tokens = action_tokens[:action_model.n_leaves_]
desire_tokens = desire_tokens[:desire_model.n_leaves_]

action_tree = build_verb_tree(action_model, action_tokens)
target_action_verbs = check_for_labels(action_tokens, target_action_verbs)
action_leaves = get_branch_leaves(action_tree, target_action_verbs)
print('Identified sub-tree')

action_sub_tree_mask = asarray([label in action_leaves for label in action_tokens])
action_sub_tree_vectors = reduced_action[action_sub_tree_mask]

num = 100
sub_tree_model, reduced_sub_action = reduce_and_cluster(action_sub_tree_vectors, num_verbs=num)
plot_dendrogram(sub_tree_model, action_tokens[action_sub_tree_mask][:num], 'Action sub-tree dendrogram', figsize=(15, 8))

desire_verb_index = 1
action_verb_index = 2
action_verb_mask = intent_frames[:, action_verb_index]

print(action_leaves)

corrected_mask = initial_mask.copy()
within_action_mask = asarray([verb in action_leaves for verb in action_verb_mask])

print('Number of documents protected', sum(within_action_mask))

# Set positive intent docs with verbs outside the action verbs to non-intent
corrected_mask[
    all([logical_not(within_action_mask), initial_mask == 1], axis=0)
] = 0

print('Num changes', sum(initial_mask != corrected_mask), 'of', sum(initial_mask == 1))

show()
