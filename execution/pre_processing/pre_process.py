# Some code to give a quick look at some of the data
from data.accessors import twitter_24k_accessor, twitter_24k_mutator, twitter_100k_accessor, twitter_100k_mutator, \
    kaggle_accessor, kaggle_mutator
from utilities.pre_processing import count_upper, process_documents, original_length, generate_header, count_emojis, \
    pull_hyperlinks, split_hashtags, manage_special_characters, count_express, count_punctuation, count_digits, \
    remove_spaces, run_partial_clean
from utilities.data_management import make_path, check_existence, check_writable

runs = [False, True]

# Generate path
data_sets = [
    {
        'data_set': '24k-abusive-tweets',
        'accessor': twitter_24k_accessor,
        'mutator': twitter_24k_mutator
    },
    {
        'data_set': '100k-abusive-tweets',
        'accessor': twitter_100k_accessor,
        'mutator': twitter_100k_mutator
    },
    {
        'data_set': 'kaggle',
        'accessor': kaggle_accessor,
        'mutator': kaggle_mutator
    }
]
base_directory = make_path('../../data/')
source_directory = base_directory / 'datasets'
dest_directory = base_directory / 'prepared_data'

check_writable(dest_directory)
for data_set in data_sets:
    set_name = data_set['data_set']
    check_existence(source_directory / set_name / (set_name + '.csv'))

# Defined pre-processing to be applied
pre_processes = [
    original_length,
    count_emojis,
    split_hashtags,
    pull_hyperlinks,
    manage_special_characters,
    count_upper,
    count_express,
    count_punctuation,
    count_digits,
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
    remove_spaces
]

for run_partial_process in runs:
    run_name = 'partial' if run_partial_process else 'pre'
    print('\nRunning', run_name, 'process.')

    processes = partial_processes if run_partial_process else pre_processes

    modified_header = generate_header(processes)

    options = {
        # 'max_documents': 1000
    }

    for data_set in data_sets:
        set_name = data_set['data_set']
        mod = '_partial' if run_partial_process else ''
        source_path = source_directory / set_name / (set_name + '.csv')
        dest_path = dest_directory / (set_name + mod + '.csv')

        process_documents(source_path, dest_path, processes, data_set['accessor'],
                          data_set['mutator'], modified_header, options)

        print(set_name, 'done.')