from data.accessors import *
from utilities.pre_processing import *
from utilities.data_management import make_path, check_existence, check_writable, open_w_pandas, expand_csv_row_size
from pandas import concat, isna
from numpy.random import permutation
from numpy import arange, savetxt

runs = [False, True]
expand_csv_row_size()

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
        'data_set': 'storm-front',
        'accessor': stormfront_accessor,
        'mutator': stormfront_mutator
    },
    {
        'data_set': 'hannah_data',
        'accessor': hannah_data_accessor,
        'mutator': hannah_data_mutator
    }
    # {
    #     'data_set': 'insults',
    #     'accessor': insults_accessor,
    #     'mutator': insults_mutator
    # }
]
base_directory = make_path('../../data/')
source_directory = base_directory / 'datasets'
dest_directory = base_directory / 'prepared_data'

check_writable(dest_directory)
for data_set in data_sets:
    set_name = data_set['data_set']
    check_existence(source_directory / set_name / (set_name + '.csv'))

partial_processes = [
    original_length,
    remove_quotes,
    manage_special_characters,
    pull_hyperlinks,
    count_tags,
    count_images,
    count_bracket_text,
    count_emojis,
    count_handles,
    split_hashtags,
    count_upper,
    run_partial_clean,
]
# Defined pre-processing to be applied
pre_processes = [
    original_length,
    remove_quotes,
    manage_special_characters,
    pull_hyperlinks,
    count_tags,
    count_images,
    count_bracket_text,
    count_emojis,
    count_handles,
    split_hashtags,
    count_upper,
    count_express,
    count_punctuation,
    count_digits,
    count_repeat_instances,
    remove_spaces,
]

# Pre process datasets
for run_partial_process in runs:
    run_name = 'partial' if run_partial_process else 'pre'
    print('\nRunning', run_name, 'process.')

    processes = partial_processes if run_partial_process else pre_processes

    modified_header = generate_header(processes)

    options = {
        # 'max_documents': 100,
        # 'encoding': 'utf8'
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

datasets = ['24k-abusive-tweets', 'kaggle', '100k-abusive-tweets']

variants = ['', '_partial']
index_map = None
for variant in variants:
    filename = dest_directory / ('mixed_redef' + variant + '.csv')
    if filename.exists():
        print('Skipping mixed')
        break

    mixed_dataset = concat(
        [open_w_pandas(dest_directory / (dataset + variant + '.csv')) for dataset in datasets]
    )

    if index_map is None:
        index_map = permutation(arange(mixed_dataset.shape[0]))
        savetxt(dest_directory / 'mixed_redef_map.csv', index_map, delimiter=',', fmt='%d')
    mixed_dataset = mixed_dataset.reset_index(drop=True).iloc[index_map].reset_index(drop=True)

    bad_indexes = mixed_dataset.index[isna(mixed_dataset['document_content'])]
    content = mixed_dataset['document_content']
    for index in bad_indexes:
        mixed_dataset.at[index, 'document_content'] = ' '

    mixed_dataset.to_csv(filename)
