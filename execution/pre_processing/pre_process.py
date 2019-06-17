# Some code to give a quick look at some of the data
from data.accessors import twitter_24k_accessor, twitter_24k_mutator, twitter_100k_accessor, twitter_100k_mutator, \
    kaggle_accessor, kaggle_mutator, insults_accessor, insults_mutator
from utilities.pre_processing import count_upper, process_documents, original_length, generate_header, count_emojis, \
    pull_hyperlinks, split_hashtags, manage_special_characters, count_express, count_punctuation, count_digits, \
    remove_spaces, run_partial_clean, count_images, count_handles
from utilities.data_management import make_path, check_existence, check_writable, open_w_pandas
from pandas import concat, isna
from numpy import where

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
    },
    {
        'data_set': 'insults',
        'accessor': insults_accessor,
        'mutator': insults_mutator
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
    count_images,
    count_emojis,
    count_handles,
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
    count_images,
    count_emojis,
    count_handles,
    split_hashtags,
    pull_hyperlinks,
    manage_special_characters,
    count_upper,
    run_partial_clean,
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

    for data_set in data_sets:
        set_name = data_set['data_set']
        mod = '_partial' if run_partial_process else ''
        source_path = source_directory / set_name / (set_name + '.csv')
        dest_path = dest_directory / (set_name + mod + '.csv')

        if dest_path.exists():
            print('Skipping', set_name, mod)
            continue

        process_documents(source_path, dest_path, processes, data_set['accessor'],
                          data_set['mutator'], modified_header, options)

        print(set_name, 'done.')


# Generate mixed dataset
print('\nGenerating mixed dataset')

datasets = ['24k-abusive-tweets', 'kaggle', 'insults']

variants = ['', '_partial']
for variant in variants:
    loaded_datasets = [open_w_pandas(dest_directory / (dataset + variant + '.csv')) for dataset in datasets]
    mixed_dataset = concat(loaded_datasets).sample(frac=1).reset_index(drop=True)

    bad_indexes = isna(mixed_dataset['document_content'])
    content = mixed_dataset['document_content']
    content = where(bad_indexes, '', content)

    mixed_dataset.to_csv(dest_directory / ('mixed_dataset' + variant + '.csv'))
