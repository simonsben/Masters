from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.initializers import Constant
from keras.models import Sequential


def generate_intent_network(embedding_matrix, max_tokens, summary=False):
    """ Returns the compiled production intent network """
    num_embeddings = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]
    attention_size = int(max_tokens / 2)

    deep_model = Sequential([
        Embedding(
            num_embeddings, embedding_dim, embeddings_initializer=Constant(embedding_matrix),
            input_length=max_tokens, trainable=False, mask_zero=True, name=('embedding_' + str(num_embeddings))
        ),
        Bidirectional(
            LSTM(max_tokens, dropout=.4, recurrent_dropout=.4, input_shape=(max_tokens, embedding_dim),
                 name='intent_bi_lstm'),
            name='intent_bi'
        ),
        Dense(attention_size, name='intent_hidden_dense'),
        Dense(1, activation='sigmoid', name='intent_prediction_dense')
    ])

    if summary:
        print('Model\n', deep_model.summary())

    deep_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return deep_model
