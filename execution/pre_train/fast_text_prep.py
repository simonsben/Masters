from utilities.data_management import move_to_root, make_path, check_writable, check_existence, open_w_pandas, \
    load_execution_params

move_to_root()

data_name = load_execution_params()['dataset']
data_path = make_path('data/prepared_data/') / (data_name + '.csv')
dest_path = make_path('.').absolute().parent / 'fastText' / 'data/' / (data_name + '.csv')

check_existence(data_path)
check_writable(dest_path)

open_w_pandas(data_path)['document_content'].to_csv(dest_path, index=False, header=False)
