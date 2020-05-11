from model.analysis import compute_abusive_intent
from keras.layers import Input, Bidirectional, LSTM, Dense, TimeDistributed, Embedding
from model.layers.attention import AttentionWithContext
from model.layers.realtime_embedding import RealtimeEmbedding
from keras.models import Model
from keras.initializers import Constant
from config import execute_verbosity


def predict_abusive_intent(realtime_documents, abusive_intent_network, method='product'):
    """
    Makes abusive intent predictions for a list of pre-processed documents

    :param RealtimeEmbedding realtime_documents: list or array of pre-processed documents
    :param Model abusive_intent_network: keras network trained to predict abuse and intent
    :param str method: method used to make abusive intent predictions
    :return tuple: tuple of abuse, intent, and abusive-intent predictions
    """
    abuse_predictions, intent_predictions = [
        predictions.reshape(-1) for predictions in abusive_intent_network.predict_generator(
            realtime_documents, verbose=execute_verbosity
        )
     ]

    abusive_intent_predictions = compute_abusive_intent(intent_predictions, abuse_predictions, method)

    return abuse_predictions, intent_predictions, abusive_intent_predictions


def generate_abusive_intent_network(max_tokens, embedding_dimension=None, embedding_matrix=None):
    # Check if either the dimension or the embeddings or embeddings are provided.
    if embedding_dimension is None and embedding_matrix is None:
        raise AttributeError('Must provide either dimension of embedding or pre-computed embeddings')

    # Check if producing training or production model
    is_production = embedding_matrix is None
    embedding_dimension = embedding_dimension if is_production else embedding_matrix.shape[1]
    attention_size = int(max_tokens / 2)

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
    abuse_time = TimeDistributed(
        Dense(attention_size, name='abuse_time_dense'),
        name='abuse_time'
    )(abuse_bi)
    abuse_attention = AttentionWithContext(name='abuse_attention')(abuse_time)
    abuse_dense = Dense(50, name='abuse_hidden_dense')(abuse_attention)
    abuse_prediction = Dense(1, activation='sigmoid', name='abuse_prediction_dense')(abuse_dense)

    # Intent
    intent_bi = Bidirectional(
        LSTM(max_tokens, dropout=.4, recurrent_dropout=.4, name='intent_bi_lstm'),
        name='intent_bi')(core_input)
    intent_dense = Dense(attention_size, name='intent_hidden_dense')(intent_bi)
    intent_prediction = Dense(1, activation='sigmoid', name='intent_prediction_dense')(intent_dense)

    abusive_intent_network = Model(
        inputs=network_input, outputs=[abuse_prediction, intent_prediction], name='abuse_intent_network'
    )

    return abusive_intent_network
