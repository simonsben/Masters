from utilities.data_management import open_w_pandas, make_path, check_existence, move_to_root, load_execution_params

# Define constants
move_to_root(4)
target_word = 'bitch'

params = load_execution_params()
data_name = params['dataset']
target_dir = make_path('data/processed_data/') / data_name / 'analysis' / 'embedding_neighbours' / target_word
target_path = target_dir / 'data.csv'

check_existence(target_dir)
check_existence(target_path)

target_data = open_w_pandas(target_path)

