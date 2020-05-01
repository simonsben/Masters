from keras.layers import Embedding, Bidirectional, LSTM, Dense, InputLayer, Flatten, Input
# from tensorflow.keras.layers import Embedding, LSTM, Dense, Flatten, Input
from keras.initializers import Constant
from keras.models import Sequential, Model
from keras_self_attention import SeqSelfAttention


# def get_core_attention_layers(max_tokens):
#     """
#     Generates core layers for intent network
#     :param max_tokens: Maximum tokens for input sequence
#     :return: Core model layers
#     """
#     attention_size = int(max_tokens / 2)
#     dense_size = int(attention_size / 2)
#
#     core_layers = [
#         LSTM(max_tokens, return_sequences=True),
#         SelfAttention(attention_size, model_api='sequential'),
#         Flatten(),
#         Dense(dense_size, name='intent_hidden_dense'),
#         Dense(1, activation='sigmoid', name='intent_prediction_dense')
#     ]
#
#     return core_layers


def generate_attention_network(max_tokens, embedding_dimension=None, embedding_matrix=None):
    """
    Generates attention network
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
    # model_layers = get_core_attention_layers(max_tokens)
    num_embeddings = embedding_matrix.shape[0]
    attention_size = int(max_tokens / 2)
    dense_size = int(attention_size / 2)

    # If training model, add embedding layer to start of model
    intent_model = Sequential([
        Embedding(
            num_embeddings, embedding_dimension, embeddings_initializer=Constant(embedding_matrix),
            input_length=max_tokens, trainable=False, name=('embedding_' + str(num_embeddings))
        ),
        LSTM(max_tokens, return_sequences=True),
        SeqSelfAttention(),
        Dense(dense_size, name='intent_hidden_dense'),
        Flatten(),
        Dense(1, activation='sigmoid', name='intent_prediction_dense')
    ])
    # model_in = Input(shape=(max_tokens,))

    # if is_production:
    #     model_layers.insert(0, )
    # else:
    #
    #     embedding_layer =
    #     model_layers.insert(0, embedding_layer)

    # Generate and compile intent model
    # intent_model = Sequential(model_layers, name='intent_network')
    # intent_model = Model(inputs=[emb], outputs=[pred])
    intent_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return intent_model
