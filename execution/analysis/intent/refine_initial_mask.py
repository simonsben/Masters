from utilities.data_management import move_to_root, make_path, load_execution_params, load_vector, open_w_pandas, \
    check_existence, read_csv
from model.analysis import intent_verb_filename
from numpy import asarray, logical_not, all, sum
from utilities.plotting import plot_dendrogram, show
from model.expansion.verb_tree import build_tree_and_collect_leaves, get_sub_tree
from model.expansion.verb_space import get_cube_mask, get_cone_mask

target_action_labels = ['kill', 'fight', 'act', 'take']
target_desire_labels = ['want', 'need', 'going', 'have']

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

max_verbs = None

cube_action_tokens = get_cube_mask(action, target_action_labels)
cube_desire_tokens = get_cube_mask(desire, target_desire_labels)

cone_action_tokens = get_cone_mask(action, target_action_labels)
cone_desire_tokens = get_cone_mask(desire, target_desire_labels)

print('Tree')
action_model, action_leaves, action_tokens, action_vectors = build_tree_and_collect_leaves(action, target_action_labels, max_labels=max_verbs)
print(action_leaves)

desire_model, desire_leaves, desire_tokens, desire_vectors = build_tree_and_collect_leaves(desire, target_desire_labels, max_labels=max_verbs)
print(desire_leaves)

print('Cube')
print(cube_action_tokens)
print(cube_desire_tokens)

print('Cone')
print(cone_action_tokens)
print(cone_desire_tokens)

# print('Dropped desire verbs', set(desire_tokens) - desire_leaves)
#
# sub_dimensions = 100
# sub_action_model, sub_action_mask = get_sub_tree(action_leaves, action_tokens, action_vectors)
# sub_desire_model, sub_desire_mask = get_sub_tree(desire_leaves, desire_tokens, desire_vectors)
#
# desire_verb_index = 1
# action_verb_index = 2
# action_verb_mask = intent_frames[:, action_verb_index]
# desire_verb_mask = intent_frames[:, desire_verb_index]
#
# corrected_mask = initial_mask.copy()
# within_action_mask = asarray([verb in action_leaves or verb == 'None' for verb in action_verb_mask])
# within_desire_mask = asarray([verb in desire_leaves or verb == 'None' for verb in desire_verb_mask])
#
# action_corrections = all([logical_not(within_action_mask), initial_mask == 1], axis=0)
# desire_corrections = all([logical_not(within_desire_mask), initial_mask == 1], axis=0)
#
# total = sum(initial_mask == 1)
# print('Desire hits', total - sum(action_corrections), 'of', total)
# print('Action hits', total - sum(desire_corrections), 'of', total)
#
# # Set positive intent docs with verbs outside the action verbs to non-intent
# # corrected_mask[
# #     all([logical_not(within_action_mask), initial_mask == 1], axis=0)
# # ] = 0
# #
# # print('Num changes', sum(initial_mask != corrected_mask), 'of', sum(initial_mask == 1))
#
#
# plot_dendrogram(sub_action_model, action_tokens[sub_action_mask][:sub_dimensions], 'Action sub-tree dendrogram', figsize=(15, 8))
# plot_dendrogram(sub_desire_model, desire_tokens[sub_desire_mask][:sub_dimensions], 'Desire sub-tree dendrogram', figsize=(15, 8))
#
# show()
