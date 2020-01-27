from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.initializers import Constant
from keras.models import Sequential


def generate_intent_network(embedding_matrix, max_tokens, summary=False):
    """ Returns the compiled intent network """
    deep_model = Sequential([
        Embedding(
            embedding_matrix.shape[0], embedding_matrix.shape[1], embeddings_initializer=Constant(embedding_matrix),
            input_length=max_tokens, trainable=False, mask_zero=True
        ),
        Bidirectional(
            LSTM(max_tokens, dropout=.4, recurrent_dropout=.4)
        ),
        Dense(50),
        Dense(1, activation='sigmoid')
    ])

    if summary:
        print('Model\n', deep_model.summary())

    deep_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return deep_model
