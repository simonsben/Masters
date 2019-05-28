from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras.callbacks import EarlyStopping


def generate_deep_model(summary=False):
    """ Returns the compiled BiLSTM model """
    deep_model = Sequential([
        Bidirectional(LSTM(
            100,
            dropout=.3,
            recurrent_dropout=.3
        ),
            input_shape=(50, 300)
        ),
        Dense(1, activation='sigmoid')
    ])
    if summary:
        print('Model\n', deep_model.summary())

    deep_model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    return deep_model


def train_deep_model(model, train, train_label):
    """ Trains the model with the passed training data """
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
    history = model.fit(
        train, train_label,
        epochs=50, verbose=1,
        callbacks=[early_stopping],
        validation_split=.2, shuffle=True, batch_size=512
    ).history

    return history
