from spacy import load
from utilities.data_management import open_w_pandas, move_to_root, make_path, check_existence, check_writable, load_execution_params
from model.extraction import filter_to_save
from time import time

move_to_root()

# Load params
params = load_execution_params()
dataset_name = params['dataset']

# Define paths
data_path = make_path('data/prepared_data/') / (dataset_name + '.csv')
dest_path = make_path('data/processed_data/') / dataset_name / 'derived' / 'spacy_parse.hdf'

# Check paths
check_existence(data_path)
check_writable(dest_path)

# Load model and data
data = open_w_pandas(data_path)
model = load('en_core_web_sm')
print('Data and model loaded, starting')

# Parse
start = time()
parsed = filter_to_save(data['document_content'].apply(model).values)
print('Parsed in', time() - start)

# Save parsed
parsed.to_hdf(dest_path, key='df', complib='blosc')
