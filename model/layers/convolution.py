from utilities.data_management import load_dataset_params
from keras import Sequential
from keras.layers import Bidirectional, TimeDistributed, LSTM, Dense, Conv2D, MaxPooling2D, Flatten
# from model.layers.attention import AttentionWithContext
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD


def generate_cnn(summary=False):
    params = load_dataset_params()
    max_tokens = params['max_document_tokens']
    fast_text_dim = params['fast_text_dim']

    deep_model = Sequential([
        TimeDistributed(
            Dense(fast_text_dim),
            input_shape=(max_tokens, fast_text_dim)
        ),
        Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(max_tokens, activation='relu'),
        Dense(1, activation='softmax')
    ])

    if summary:
        print('Model\n', deep_model.summary())

    deep_model.compile(categorical_crossentropy, SGD(lr=.01), metrics=['accuracy'])

    return deep_model
