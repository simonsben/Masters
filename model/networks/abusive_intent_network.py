from utilities.data_management.model_management import load_model_weights
from utilities.data_management import get_embedding_path, get_model_path
from utilities.pre_processing import runtime_clean
from keras.layers import Input, Bidirectional, LSTM, Dense, TimeDistributed, Embedding, Multiply
from model.layers.attention import AttentionWithContext
from model.layers.realtime_embedding import RealtimeEmbedding
from keras.models import Model
from keras.initializers import Constant
from fasttext import load_model
from numpy import hstack, ndarray
from config import execute_verbosity, max_tokens, embedding_dimension


def predict_abusive_intent(raw_documents, network=None, return_model=False):
    """
    Makes abusive intent predictions for a list of pre-processed documents

    :param ndarray raw_documents: array of pre-processed documents
    :param Model network: keras network trained to predict abuse and intent
    :param bool return_model: Whether to return the model as well as the predictions
    :return tuple: tuple of abuse, intent, and abusive-intent predictions
    """
    if network is None:
        embedding_path = get_embedding_path()
        intent_path = get_model_path('intent')
        abuse_path = get_model_path('abuse')

        documents = runtime_clean(raw_documents)
        embedding_model = load_model(embedding_path)
        raw_documents = RealtimeEmbedding(embedding_model, documents)
        print('Loaded embeddings')

        network = generate_abusive_intent_network(max_tokens, embedding_dimension=embedding_dimension)
        load_model_weights(network, intent_path)
        load_model_weights(network, abuse_path)
        print(network.summary())

    predictions = hstack(network.predict_generator(raw_documents, verbose=execute_verbosity)).transpose()
    if return_model:
        return network, predictions
    return predictions


def generate_abusive_intent_network(max_tokens, embedding_dimension=None, embedding_matrix=None):
    # Check if either the dimension or the embeddings or embeddings are provided.
    if embedding_dimension is None and embedding_matrix is None:
        raise AttributeError('Must provide either dimension of embedding or pre-computed embeddings')

    # Check if producing training or production model
    is_production = embedding_matrix is None
    embedding_dimension = embedding_dimension if is_production else embedding_matrix.shape[1]
    attention_size = int(max_tokens / 2)
    final_dense_size = 50

    # Define network
    input_shape = (max_tokens, embedding_dimension) if is_production else (max_tokens,)
    core_input = network_input = Input(shape=input_shape)

    # If not generating for production, add embedding layer
    if not is_production:
        num_embeddings = embedding_matrix.shape[0]

        core_input = Embedding(
            num_embeddings, embedding_dimension, embeddings_initializer=Constant(embedding_matrix),
            input_length=max_tokens, trainable=False, mask_zero=True, name=('embedding_' + str(num_embeddings))
        )(network_input)

    # Abuse
    abuse_bi = Bidirectional(
        LSTM(max_tokens, dropout=.5, recurrent_dropout=.5, return_sequences=True, name='abuse_bi_lstm'),
        name='abuse_bi'
    )(core_input)
    abuse_attention = AttentionWithContext(name='abuse_attention')(abuse_bi)
    abuse_dense = Dense(final_dense_size, name='abuse_hidden_dense')(abuse_attention)
    abuse_prediction = Dense(1, activation='sigmoid', name='abuse_prediction_dense')(abuse_dense)

    # Intent
    intent_bi = Bidirectional(
        LSTM(max_tokens, dropout=.5, recurrent_dropout=.5, return_sequences=True, name='intent_bi_lstm'),
        name='intent_bi')(core_input)
    intent_attention = AttentionWithContext(name='intent_attention')(intent_bi)
    intent_dense = Dense(final_dense_size, name='intent_hidden_dense')(intent_attention)
    intent_prediction = Dense(1, activation='sigmoid', name='intent_prediction_dense')(intent_dense)

    abusive_intent_prediction = Multiply(name='abusive_intent_prediction')([abuse_prediction, intent_prediction])

    abusive_intent_network = Model(
        inputs=network_input, outputs=[abuse_prediction, intent_prediction, abusive_intent_prediction],
        name='abuse_intent_network'
    )

    return abusive_intent_network
