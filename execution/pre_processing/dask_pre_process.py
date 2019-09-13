from data.accessors import stormfront
from utilities.pre_processing import count_upper, dask_process_documents, original_length, count_emojis, \
    pull_hyperlinks, split_hashtags, manage_special_characters, count_express, count_punctuation, count_digits, \
    remove_spaces, run_partial_clean, count_images, count_handles, count_repeat_instances, count_tags
from utilities.data_management import make_path, check_existence, check_writable, open_w_pandas, move_to_root, make_dir
from pandas import concat, isna
from csv import field_size_limit
from sys import maxsize

move_to_root()

runs = [False, True]

# Enable larger field sizes
max_size = maxsize
while True:
    try:
        field_size_limit(max_size)
        break
    except OverflowError:
        max_size = int(max_size/10)

# Generate path
data_sets = [
    {
        'data_set': 'storm-front',
        'constants': stormfront
    }
]

base_directory = make_path('data/')
source_directory = base_directory / 'datasets'
dest_directory = base_directory / 'prepared_data'
tmp_directory = dest_directory / 'tmp'

make_dir(tmp_directory)
check_writable(dest_directory)
for data_set in data_sets:
    set_name = data_set['data_set']
    check_existence(source_directory / set_name / (set_name + '.csv'))

# Defined pre-processing to be applied
pre_processes = [
    original_length,
    count_tags,
    count_images,
    pull_hyperlinks,
    count_emojis,
    count_handles,
    split_hashtags,
    manage_special_characters,
    count_upper,
    count_express,
    count_punctuation,
    count_digits,
    count_repeat_instances,
    remove_spaces
]
partial_processes = [
    original_length,
    count_tags,
    count_images,
    pull_hyperlinks,
    count_emojis,
    count_handles,
    split_hashtags,
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

    options = {
        # 'max_documents': 10000,
        'encoding': 'utf8',
        'header': None
    }

    for data_set in data_sets:
        set_name = data_set['data_set']
        mod = '_partial' if run_partial_process else ''
        source_path = source_directory / set_name / (set_name + '.csv')
        dest_path = dest_directory / (set_name + mod + '.csv')

        # if dest_path.exists():
        #     print('Skipping', set_name, mod)
        #     continue

        dest_path = tmp_directory / (set_name + mod + '.*.csv')
        dask_process_documents(source_path, dest_path, processes, data_set['constants'], options)

        print(set_name, 'done.')


# Generate mixed dataset
print('\nGenerating mixed dataset')

datasets = ['24k-abusive-tweets', 'kaggle', '100k-abusive-tweets']

variants = ['', '_partial']
for variant in variants:
    filename = dest_directory / ('mixed_redef' + variant + '.csv')
    if filename.exists():
        print('Skipping mixed', variant)
        continue

    mixed_dataset = concat(
        [open_w_pandas(dest_directory / (dataset + variant + '.csv')) for dataset in datasets]
    ).sample(frac=1).reset_index(drop=True)

    bad_indexes = mixed_dataset.index[isna(mixed_dataset['document_content'])]
    content = mixed_dataset['document_content']
    for index in bad_indexes:
        mixed_dataset.at[index, 'document_content'] = ' '

    mixed_dataset.to_csv(filename)
