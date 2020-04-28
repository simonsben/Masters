from keras.layers import Embedding, Bidirectional, LSTM, Dense, InputLayer, Conv1D, Flatten, MaxPooling1D
from keras.initializers import Constant
from keras.models import Sequential


def get_core_convolution_layers(max_tokens):
    """
    Generates core layers for intent network
    :param max_tokens: Maximum tokens for input sequence
    :return: Core model layers
    """
    attention_size = int(max_tokens / 2)

    core_layers = [
        Conv1D(138, 5, activation='relu'),
        MaxPooling1D(5),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(15),
        # Conv1D(128, 5, activation='relu'),
        # MaxPooling1D(15),
        Flatten(),
        Dense(attention_size, name='intent_hidden_dense'),
        Dense(1, activation='sigmoid', name='intent_prediction_dense')
    ]

    return core_layers


def generate_convolution_network(max_tokens, embedding_dimension=None, embedding_matrix=None):
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
    model_layers = get_core_convolution_layers(max_tokens)

    # If training model, add embedding layer to start of model
    if is_production:
        model_layers.insert(0, InputLayer(input_shape=(max_tokens, embedding_dimension)))
    else:
        num_embeddings = embedding_matrix.shape[0]

        embedding_layer = Embedding(
            num_embeddings, embedding_dimension, embeddings_initializer=Constant(embedding_matrix),
            input_length=max_tokens, trainable=False, name=('embedding_' + str(num_embeddings))
        )
        model_layers.insert(0, embedding_layer)

    # Generate and compile intent model
    intent_model = Sequential(model_layers, name='intent_network')
    intent_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return intent_model
