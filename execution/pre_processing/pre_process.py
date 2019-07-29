from data.accessors import twitter_24k_accessor, twitter_24k_mutator, twitter_100k_accessor, twitter_100k_mutator, \
    kaggle_accessor, kaggle_mutator, insults_accessor, insults_mutator, stormfront_accessor, stormfront_mutator
from utilities.pre_processing import count_upper, process_documents, original_length, generate_header, count_emojis, \
    pull_hyperlinks, split_hashtags, manage_special_characters, count_express, count_punctuation, count_digits, \
    remove_spaces, run_partial_clean, count_images, count_handles, count_repeat_instances, count_tags
from utilities.data_management import make_path, check_existence, check_writable, open_w_pandas, expand_csv_row_size
from pandas import concat, isna
from numpy.random import permutation
from numpy import arange, savetxt

runs = [False, True]
expand_csv_row_size()

# Generate path
data_sets = [
    # {
    #     'data_set': '24k-abusive-tweets',
    #     'accessor': twitter_24k_accessor,
    #     'mutator': twitter_24k_mutator
    # },
    # {
    #     'data_set': '100k-abusive-tweets',
    #     'accessor': twitter_100k_accessor,
    #     'mutator': twitter_100k_mutator
    # },
    {
        'data_set': 'kaggle',
        'accessor': kaggle_accessor,
        'mutator': kaggle_mutator
    },
    # {
    #     'data_set': 'storm-front',
    #     'accessor': stormfront_accessor,
    #     'mutator': stormfront_mutator
    # },
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

# Defined pre-processing to be applied
pre_processes = [
    original_length,
    count_tags,
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
    count_repeat_instances,
    remove_spaces
]
partial_processes = [
    original_length,
    count_tags,
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
        # 'max_documents': 10000,
        'encoding': 'utf8'
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
        savetxt(dest_directory / 'mixed_redef_map.csv', index_map, delimiter=',')
    mixed_dataset = mixed_dataset.reset_index(drop=True).reindex(index_map).sort_index()

    bad_indexes = mixed_dataset.index[isna(mixed_dataset['document_content'])]
    content = mixed_dataset['document_content']
    for index in bad_indexes:
        mixed_dataset.at[index, 'document_content'] = ' '

    mixed_dataset.to_csv(filename)
