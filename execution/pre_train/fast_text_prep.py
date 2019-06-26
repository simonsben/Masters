from utilities.data_management import open_w_pandas, move_to_root, make_path, check_writable, check_existence, \
    load_execution_params

move_to_root()

data_name = load_execution_params()['dataset']
data_path = make_path('data/prepared_data/') / (data_name + '.csv')
dest_path = make_path('../fastText/data/') / (data_name + '.csv')

check_existence(data_path)
check_writable(dest_path)

data = open_w_pandas(data_path)
data['document_content'].to_csv(dest_path, index=False, header=False)

