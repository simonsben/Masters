from keras import Sequential
from keras.layers import Bidirectional, LSTM, Dense, TimeDistributed
from model.layers.attention import AttentionWithContext


def generate_production_abuse_network(max_tokens, embedding_dim, summary=False):
    """ Returns the compiled abuse network """
    # Define network
    deep_model = Sequential([
        Bidirectional(
            LSTM(embedding_dim, dropout=.5, recurrent_dropout=.5, return_sequences=True, name='abuse_bi_lstm'),
            input_shape=(max_tokens, embedding_dim), name='abuse_bi'
        ),
        TimeDistributed(
            Dense(150, name='abuse_time_dense'),
            name='abuse_time'
        ),
        AttentionWithContext(name='abuse_attention'),
        Dense(50, name='abuse_hidden_dense'),
        Dense(1, activation='sigmoid', name='abuse_prediction_dense')
    ])

    if summary:
        print('Model\n', deep_model.summary())

    deep_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return deep_model
