from model.networks import generate_abusive_intent_network
from utilities.data_management import get_model_path, make_dir
from config import dataset, max_tokens, embedding_dimension

# Define file paths
intent_weights_path = get_model_path('intent')
abuse_weights_path = get_model_path('abuse')
production_dir = intent_weights_path.parent / 'production/'

make_dir(production_dir)

# Generate abuse model and load trained weights
production_network = generate_abusive_intent_network(max_tokens, embedding_dimension=embedding_dimension)
production_network.load_weights(intent_weights_path, by_name=True)
production_network.load_weights(abuse_weights_path, by_name=True)
print(production_network.summary())

production_network.save(production_dir)
print('Model exported.')
