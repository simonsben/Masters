from keras import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Embedding, TimeDistributed
from keras.initializers import Constant
from model.networks.attention import AttentionWithContext
from utilities.data_management import load_dataset_params


def generate_abuse_network(embedding_matrix, max_tokens, summary=False):
    """ Returns the compiled abuse network """
    params = load_dataset_params()
    # max_tokens = params['max_document_tokens']
    fast_text_dim = params['fast_text_dim']

    # Define network
    deep_model = Sequential([
        Embedding(embedding_matrix.shape[0], fast_text_dim,
                  embeddings_initializer=Constant(embedding_matrix),
                  input_length=max_tokens, trainable=False, mask_zero=True
                  ),
        Bidirectional(
            LSTM(int(fast_text_dim / 2), dropout=.5, recurrent_dropout=.5,
                 return_sequences=True)
        ),
        TimeDistributed(
            Dense(150)
        ),
        AttentionWithContext(),
        Dense(50),
        Dense(1, activation='sigmoid')
    ])

    if summary:
        print('Model\n', deep_model.summary())

    deep_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return deep_model
