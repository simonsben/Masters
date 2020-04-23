from model.networks import generate_intent_network, generate_abuse_network
from utilities.data_management import get_model_path, make_path, freeze_session
from tensorflow.train import write_graph
import config

# Load config
dataset = config.dataset
max_tokens = config.max_tokens
embedding_dimension = config.embedding_dimension

# Define file paths
intent_weights_path = get_model_path('intent')
abuse_weights_path = get_model_path('abuse')
production_dir = make_path('data/models') / dataset / 'analysis'

# Generate intent model and load trained weights
production_network = generate_intent_network(max_tokens, embedding_dimension=embedding_dimension)
production_network.load_weights(intent_weights_path, by_name=True)
print(production_network.summary())

# Convert Keras model to Tensorflow graph and save
graph = freeze_session(production_network)
write_graph(graph, str(production_dir), 'intent.pb')
print('Intent model exported.')


# Generate abuse model and load trained weights
production_network = generate_abuse_network(max_tokens, embedding_dimension=embedding_dimension)
production_network.load_weights(abuse_weights_path, by_name=True)
print(production_network.summary())

# Convert Keras model to Tensorflow graph and save
graph = freeze_session(production_network)
write_graph(graph, str(production_dir), 'abuse.pb')
print('Intent model exported.')
