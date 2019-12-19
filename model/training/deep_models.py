from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Conv2D, MaxPooling2D, Flatten, Dropout, TimeDistributed
from keras.callbacks import EarlyStopping
from utilities.data_management import load_dataset_params
from model.networks.attention import AttentionWithContext


def generate_deep_model(summary=False):
    """ Returns the compiled BiLSTM model """
    params = load_dataset_params()
    max_tokens = params['max_document_tokens']
    fast_text_dim = params['fast_text_dim']

    # Define network layers
    deep_model = Sequential([
        Bidirectional(
            LSTM(100, dropout=.3, recurrent_dropout=.3),
            input_shape=(max_tokens, fast_text_dim)
        ),
        Dense(1, activation='sigmoid')
    ])
    if summary:
        print('Model\n', deep_model.summary())

    deep_model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    return deep_model


def generate_attention_model(summary=False):
    """ Generate BiLSTM with additional Attention layer """
    params = load_dataset_params()
    max_tokens = params['max_document_tokens']
    fast_text_dim = params['fast_text_dim']

    deep_model = Sequential([
        Bidirectional(
            LSTM(int(fast_text_dim / 2), dropout=.3, recurrent_dropout=.3, return_sequences=True),
            input_shape=(max_tokens, fast_text_dim)
        ),
        TimeDistributed(
            Dense(int(fast_text_dim * 2 / 3))
        ),
        AttentionWithContext(),
        Dense(100),
        Dense(1, activation='sigmoid')
    ])

    if summary:
        print('Model\n', deep_model.summary())

    deep_model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    return deep_model


def load_attention(source_path):
    path = str(source_path)
    return load_model(path, custom_objects={'AttentionWithContext': AttentionWithContext})


def train_deep_model(model, train, train_label):
    """ Trains the model with the passed training data """

    # Define early stopping condition to prevent over-fitting
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

    # Fit model
    history = model.fit(
        train, train_label,
        epochs=50, verbose=1,
        callbacks=[early_stopping],
        validation_split=.15, shuffle=True, batch_size=512
    ).history

    return history
