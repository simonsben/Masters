# Some code to give a quick look at some of the data
from pathlib import Path
from data.accessors import twitter_24k_accessor, twitter_24k_mutator
from utilities.pre_processing import count_upper, process_documents, original_length, modified_header, count_emojis, \
    pull_hyperlinks

# Generate path
data_set = '24k-abusive-tweets'
source_filename = Path('../data/datasets') / data_set / (data_set + '.csv')
dest_filename = Path('../data/prepared_data') / (data_set + '.csv')

print('source: ' + str(source_filename))
print('destination: ' + str(dest_filename))

# Defined pre-processing to be applied
processes = [
    original_length,
    count_upper,
    count_emojis,
    pull_hyperlinks
]

options = {
    # 'max_documents': 15000
}

# Apply pre-processing
process_documents(source_filename, dest_filename, processes, twitter_24k_accessor,
                  twitter_24k_mutator, modified_header, options)
print('Done processing')
