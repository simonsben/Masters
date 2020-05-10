from model.networks import generate_intent_network, generate_convolution_network, generate_attention_network
from utilities.data_management import make_dir, make_path, open_w_pandas, open_embeddings, check_existence, \
    get_model_path, load_vector, vector_to_file
from utilities.pre_processing import runtime_clean, token_to_index
from model.training import train_term_learner, train_deep_learner
from config import dataset, fast_text_model, max_tokens, mask_refinement_method
from scipy.sparse import load_npz
from fasttext import load_model
from model.layers.realtime_embedding import RealtimeEmbedding

# Define paths
intent_weights_path = get_model_path('intent')
# embedding_path = make_path('data/prepared_lexicon/') / ('abusive_intent-' + fast_text_model + '.csv')
embedding_path = make_path('data/models/') / dataset / 'derived' / (dataset + '.bin')
base_path = make_path('data/processed_data/') / dataset / 'analysis'
intent_path = base_path / 'intent'
context_path = intent_path / 'contexts.csv'
initial_label_path = intent_path / (mask_refinement_method + '_mask.csv')
document_matrix_path = intent_path / 'document_matrix.npz'
label_path = intent_path / 'intent_training_labels.csv'
token_path = intent_path / 'ngrams.csv'

# Check for files and make directories
check_existence([embedding_path, context_path, initial_label_path, document_matrix_path, token_path])
make_dir(intent_weights_path.parent)
print('Config complete.')

# Load embeddings and contexts
# tokens, embeddings = open_embeddings(embedding_path)
embedding_model = load_model(str(embedding_path))

raw_contexts = open_w_pandas(context_path)['contexts'].values
initial_labels = load_vector(initial_label_path)
document_matrix = load_npz(document_matrix_path)
tokens = load_vector(token_path)
token_mapping = {token: index for index, token in enumerate(tokens)}
print('Loaded data.')

# Clean contexts and enumerate tokens
contexts = runtime_clean(raw_contexts)
# enumerated_contexts, token_mapping = token_to_index(contexts, tokens, return_mapping=True)
print('Prepared data')

# Generate fresh (untrained model)
labels = initial_labels.copy()
realtime = RealtimeEmbedding(embedding_model, contexts, labels)
model = generate_intent_network(max_tokens, embedding_dimension=realtime.embedding_dimension)
print('Generated model\n', model.summary())

rounds = 4  # Number of rounds of training to perform

# Run training rounds
for round_num in range(rounds):
    print('Starting full round', round_num + 1, 'of', rounds)

    # Train deep model
    model, labels, round_prediction = train_deep_learner(model, labels, realtime)

    # Run term learner
    token_labels = labels.copy()
    positive_terms, negative_terms, total_terms, token_labels = train_term_learner(token_labels, tokens, token_mapping, document_matrix)

    # Count number of documents identified by term learner
    labels = token_labels

    print('Round features')
    print(positive_terms)
    print(negative_terms)
    print(total_terms)

vector_to_file(labels, label_path)
model.save_weights(str(intent_weights_path))

