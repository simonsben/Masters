from utilities.data_management import open_w_pandas, check_existence,  make_path, check_writable, split_sets, \
    to_numpy_array, move_to_root, load_execution_params
from model.extraction import vectorize_data
from model.training import generate_deep_model, train_deep_model
from fastText import load_model
from pandas import read_pickle

move_to_root()

dataset_name = load_execution_params()['dataset']
filename = make_path('data/prepared_data/') / (dataset_name + '.csv')
fast_text_filename = make_path('data/lexicons/fast_text/fast_text.bin')
vectorized_path = make_path('data/processed_data/') / dataset_name / 'derived/' / 'fast_text.pkl'
model_filename = make_path('data/models/') / dataset_name / 'derived/' / 'fast_text.h5'

check_existence(filename)
check_existence(fast_text_filename)
check_writable(vectorized_path)
check_writable(model_filename)

if model_filename.exists():
    print('Skipping deep model')

else:
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


    # Generate and train model
    deep_model = generate_deep_model(True)
    history = train_deep_model(deep_model, train, train_label)

    print('Deep model trained.')

    deep_model.save(str(model_filename))
    print('Deep model saved.')

    print(history['acc'])
    print(history['val_acc'])
    print(history['loss'])
    print(history['val_loss'])
