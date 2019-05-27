from utilities.data_management import open_w_pandas, check_existence,  make_path, check_writable, split_sets, \
    to_numpy_array
from model.extraction import vectorize_data
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras.callbacks import EarlyStopping
from fastText import load_model
from pandas import read_pickle


filename = make_path('../../data/prepared_data/24k-abusive-tweets.csv')
fast_text_filename = make_path('../../data/lexicons/fast_text/fast_text.bin')
vectorized_directory = make_path('../../data/processed_data/')
vectorized_path = vectorized_directory / (filename.stem + '.pkl')
model_filename = make_path('../../data/models/derived/fast_text.h5')

check_existence(filename)
check_existence(fast_text_filename)
check_writable(vectorized_directory)
check_writable(model_filename)

dataset = open_w_pandas(filename)
print('Dataset loaded\n', dataset)

if not vectorized_path.exists():
    print('Vectorizing data')

    fast_text_model = load_model(str(fast_text_filename))
    print('Model loaded\n', fast_text_model)

    vectorize_data(dataset, fast_text_model)
    fast_text_model = None
    print(dataset['vectorized_content'])

    dataset['vectorized_content'].to_pickle(vectorized_path)
    print('Saved vectorized data.')
else:
    print('Loading vectorized data')
    dataset['vectorized_content'] = read_pickle(vectorized_path)

print('Vectors ready\n', dataset['vectorized_content'])


# Split training and test sets
(train, test), (train_label, test_label) \
    = split_sets(dataset['vectorized_content'], lambda doc: doc, labels=dataset['is_abusive'])
train, test, train_label, test_label = to_numpy_array([train, test, train_label, test_label])
print(train.shape, train_label.shape)

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
print('Model\n', deep_model.summary())


deep_model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
history = deep_model.fit(
    train, train_label,
    epochs=50, verbose=1,
    callbacks=[early_stopping],
    validation_split=.2, shuffle=True, batch_size=512
).history
print('Deep model trained.')

deep_model.save(str(model_filename))
print('Deep model saved.')

print(history['acc'])
print(history['val_acc'])
print(history['loss'])
print(history['val_loss'])
