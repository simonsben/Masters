from keras import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Embedding, TimeDistributed
from keras.initializers import Constant
from model.layers.attention import AttentionWithContext
from utilities.data_management import load_dataset_params


def generate_deep_model(embedding_matrix, summary=False):
    """ Returns the compiled BiLSTM model """
    params = load_dataset_params()
    max_tokens = params['max_document_tokens']
    fast_text_dim = params['fast_text_dim']

    # Define network layers
    deep_model = Sequential([
        Embedding(embedding_matrix.shape[0], fast_text_dim,
                  embeddings_initializer=Constant(embedding_matrix),
                  input_length=max_tokens, trainable=True
                  ),
        Bidirectional(
            LSTM(int(fast_text_dim / 2), dropout=.3, recurrent_dropout=.3, return_sequences=True),
            input_shape=(max_tokens, fast_text_dim)
        ),
        TimeDistributed(
            Dense(200)
        ),
        AttentionWithContext(),
        Dense(100),
        Dense(1, activation='sigmoid')
    ])

    if summary:
        print('Model\n', deep_model.summary())

    deep_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return deep_model
