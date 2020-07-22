from model.networks import generate_intent_network, generate_tree_sequence_network
from utilities.data_management import make_dir, make_path, open_w_pandas, check_existence, \
    get_model_path, load_vector, vector_to_file, get_embedding_path
from utilities.pre_processing import runtime_clean
from model.training import train_sequence_learner, train_deep_learner, get_consensus, reinforce_xgboost
from config import dataset, max_tokens, mask_refinement_method, num_training_rounds
from scipy.sparse import load_npz
from fasttext import load_model
from model.layers.realtime_embedding import RealtimeEmbedding
from numpy import sum

# Define paths
intent_weights_path = get_model_path('intent')
embedding_path = get_embedding_path()
base_path = make_path('data/processed_data/') / dataset / 'analysis'
intent_path = base_path / 'intent'
context_path = intent_path / 'contexts.csv'
initial_label_path = intent_path / (mask_refinement_method + '_mask.csv')
document_matrix_path = intent_path / 'document_matrix.npz'
label_path = intent_path / 'intent_training_labels.csv'
token_path = intent_path / 'ngrams.csv'
midway_mask_generator = lambda info: intent_path / ('midway_mask_' + str(info[0]) + '_of_' + str(info[1]) + '.csv')

# Check for files and make directories
check_existence([embedding_path, context_path, initial_label_path, document_matrix_path, token_path])
make_dir(intent_weights_path.parent)
print('Config complete.')

# Load embeddings and contexts
embedding_model = load_model(str(embedding_path))

raw_contexts = open_w_pandas(context_path)['contexts'].values
initial_labels = load_vector(initial_label_path)
document_matrix = load_npz(document_matrix_path)
tokens = load_vector(token_path)
token_mapping = {token: index for index, token in enumerate(tokens)}
print('Loaded data.')

# Clean contexts and enumerate tokens
contexts = runtime_clean(raw_contexts)
print('Prepared data')

# Generate fresh (untrained model)
labels = initial_labels.copy()

realtime = RealtimeEmbedding(embedding_model, contexts, labels)
deep_model = generate_intent_network(max_tokens, embedding_dimension=realtime.embedding_dimension)
# tree_model = generate_tree_sequence_network()
print('Generated model\n', deep_model.summary())

rounds = num_training_rounds  # Number of rounds of training to perform

# Run training rounds
for round_num in range(rounds):
    print('Starting full round', round_num + 1, 'of', rounds)

    # Run term learner
    token_labels = train_sequence_learner(labels, tokens, document_matrix)

    # Run tree sequence learner
    # tree_labels = reinforce_xgboost(tree_model, document_matrix, labels, initial_labels, features=tokens)

    # Train deep model
    deep_labels = train_deep_learner(deep_model, labels, realtime)

    # Count number of documents identified by term learner
    new_labels = get_consensus(labels, deep_labels, token_labels)

    print(sum(labels != new_labels), 'classification changes.')
    labels = new_labels

    vector_to_file(labels, midway_mask_generator((round_num, rounds)))

    # Save model each round
    vector_to_file(labels, label_path)
    deep_model.save_weights(str(get_model_path('intent', index=round_num)))
print('Model training completed.')

vector_to_file(labels, label_path)
deep_model.save_weights(str(intent_weights_path))
print('Model saved.')
