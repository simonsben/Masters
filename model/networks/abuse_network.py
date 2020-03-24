from keras import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Embedding, TimeDistributed
from keras.initializers import Constant
from model.layers.attention import AttentionWithContext


def generate_abuse_network(embedding_matrix, max_tokens, summary=False):
    """ Returns the compiled abuse network """
    fast_text_dim = embedding_matrix.shape[1]

    # Define network
    deep_model = Sequential([
        Embedding(embedding_matrix.shape[0], fast_text_dim, embeddings_initializer=Constant(embedding_matrix),
                  input_length=max_tokens, trainable=False, mask_zero=True,
                  name=('embedding_' + str(embedding_matrix.shape[0]))
                  ),
        Bidirectional(
            LSTM(int(fast_text_dim / 2), dropout=.5, recurrent_dropout=.5, return_sequences=True, name='abuse_bi_lstm'),
            name='abuse_bi'
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
