from keras.layers import Bidirectional, LSTM, Dense
from keras.models import Sequential


def generate_production_intent_network(max_tokens, embedding_dim, summary=False):
    """ Returns the compiled production intent network """
    attention_size = int(max_tokens / 2)

    deep_model = Sequential([
        Bidirectional(
            LSTM(max_tokens, dropout=.4, recurrent_dropout=.4),
            input_shape=(max_tokens, embedding_dim), name='intent_bi_lstm'
        ),
        Dense(attention_size, name='intent_hidden_dense'),
        Dense(1, activation='sigmoid', name='intent_prediction_dense')
    ])

    if summary:
        print('Model\n', deep_model.summary())

    deep_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return deep_model

