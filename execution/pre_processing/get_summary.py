# Summarizes all pre-processed documents
from utilities.data_management import check_readable, make_path, open_w_pandas
from utilities.analysis import take_basics
from os import listdir

# Define constants
dir_path = make_path('../../data/prepared_data/')
check_readable(dir_path)
files = listdir(dir_path)   # Get pre-processed files

# Summarize each file
for file in files:
    print('File: ' + file)
    file_path = dir_path / file

    dataset = open_w_pandas(file_path)
    stats, correlations = take_basics(dataset, display=True)
