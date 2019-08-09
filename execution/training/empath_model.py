from utilities.data_management import open_w_pandas, make_path, check_writable, move_to_root, load_execution_params, \
    save_prepared
from model.extraction import empath_matrix
from model.training import train_xg_boost

if __name__ == '__main__':
    move_to_root()  # Change PWD to root project directory

    # Define source files
    dataset_name = load_execution_params()['dataset']
    data_filename = make_path('data/prepared_data/') / (dataset_name + '.csv')
    processed_base = make_path('data/processed_data/') / dataset_name / 'lexicon'
    model_dir = make_path('data/models/' + dataset_name + '/lexicon/')

    # Define destination directory
    check_writable(model_dir)
    check_writable(processed_base)

    model_filename = model_dir / 'empath.bin'
    if model_filename.exists():
        print('Skipping Empath')
    else:
        print('Starting Empath')

        # Load dataset
        dataset = open_w_pandas(data_filename)
        print('Data loaded.', dataset.shape)

        # Load lexicon and construct document-term matrix
        document_matrix, features = empath_matrix(dataset)

        # Train model
        model, (train, test) = train_xg_boost(document_matrix, dataset['is_abusive'].to_numpy(), return_data=True)

        # Save model
        save_prepared(processed_base, 'empath', train[0], test[0])
        model.save_model(str(model_filename))
        print('Empath completed.')
