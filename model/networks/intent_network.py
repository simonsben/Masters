from keras.layers import Embedding, Bidirectional, LSTM, Dense, InputLayer
from keras.initializers import Constant
from keras.models import Sequential


def get_core_layers(max_tokens, embedding_dimension):
    """
    Generates core layers for intent network
    :param max_tokens: Maximum tokens for input sequence
    :param embedding_dimension: Dimension of the word embeddings
    :return: Core model layers
    """
    attention_size = int(max_tokens / 2)

    core_layers = [
        InputLayer(input_shape=(max_tokens, embedding_dimension)),
        Bidirectional(
            LSTM(max_tokens, dropout=.4, recurrent_dropout=.4, name='intent_bi_lstm'),
            name='intent_bi'
        ),
        Dense(attention_size, name='intent_hidden_dense'),
        Dense(1, activation='sigmoid', name='intent_prediction_dense')
    ]

    return core_layers


def generate_intent_network(max_tokens, embedding_dimension=None, embedding_matrix=None):
    """
    Generates intent network
    :param max_tokens: Maximum tokens for input sequence
    :param embedding_dimension: Dimension of the word embeddings [optional]
    :param embedding_matrix: Matrix of pre-computed word embeddings [optional]
    :return: Intent network
    """

    # Check if either the dimension or the embeddings or embeddings are provided.
    if embedding_dimension is None and embedding_matrix is None:
        raise AttributeError('Must provide either dimension of embedding or pre-computed embeddings')

    # Check if producing training or production model
    is_production = embedding_matrix is None
    embedding_dimension = embedding_dimension if is_production else embedding_matrix.shape[1]

    # Generate core layers
    model_layers = get_core_layers(max_tokens, embedding_dimension)

    # If training model, add embedding layer to start of model
    if not is_production:
        num_embeddings = embedding_matrix.shape[0]

        embedding_layer = Embedding(
            num_embeddings, embedding_dimension, embeddings_initializer=Constant(embedding_matrix),
            input_length=max_tokens, trainable=False, mask_zero=True, name=('embedding_' + str(num_embeddings))
        )
        model_layers.insert(1, embedding_layer)     # Input layer is index 0

    # Generate and compile intent model
    intent_model = Sequential(model_layers)
    intent_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return intent_model
