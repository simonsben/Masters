# Some code to give a quick look at some of the data
from data.accessors import twitter_24k_accessor, twitter_24k_mutator
from utilities.pre_processing import count_upper, process_documents, original_length, generate_header, count_emojis, \
    pull_hyperlinks, split_hashtags, manage_special_characters, count_express, count_punctuation, count_digits, \
    remove_spaces, run_partial_clean, count_images, count_bitch
from utilities.data_management import make_path, check_existence, check_writable, open_w_pandas

runs = [False, True]

# Generate path
data_set = {
    'data_set': '24k-abusive-tweets',
    'accessor': twitter_24k_accessor,
    'mutator': twitter_24k_mutator
}

base_directory = make_path('../../data/')
source_directory = base_directory / 'datasets'
dest_directory = base_directory / 'prepared_data'
mod_name = '_wo_bitch'

check_writable(dest_directory)
set_name = data_set['data_set']
check_existence(source_directory / set_name / (set_name + '.csv'))

# Defined pre-processing to be applied
pre_processes = [
    original_length,
    count_images,
    count_emojis,
    split_hashtags,
    pull_hyperlinks,
    manage_special_characters,
    count_upper,
    count_express,
    count_punctuation,
    count_digits,

    count_bitch,    # Alternate filter

    remove_spaces
]
partial_processes = [
    original_length,
    count_emojis,
    split_hashtags,
    pull_hyperlinks,
    manage_special_characters,
    count_upper,
    run_partial_clean,

    count_bitch,    # Alternate filter

    remove_spaces
]

# Pre process datasets
for run_partial_process in runs:
    run_name = 'partial' if run_partial_process else 'pre'
    print('\nRunning', run_name, 'process.')

    processes = partial_processes if run_partial_process else pre_processes

    modified_header = generate_header(processes)

    options = {
        # 'max_documents': 1000
    }

    mod = mod_name + ('_partial' if run_partial_process else '')
    set_name = data_set['data_set']

    source_path = source_directory / set_name / (set_name + '.csv')
    dest_path = dest_directory / (set_name + mod + '.csv')

    process_documents(source_path, dest_path, processes, data_set['accessor'],
                      data_set['mutator'], modified_header, options)
