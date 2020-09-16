from data.accessors import *
from utilities.pre_processing import *
from utilities.data_management import make_path, check_existence, check_writable, expand_csv_row_size

# Stupid pre-processing script I dont want to re-write

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
    count_acronym,
    count_digits,
    count_repeat_instances,
    remove_spaces,
    count_repeat_words,
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
    count_acronym,
    count_express,
    count_apostrophe,
    count_punctuation,
    count_digits,
    count_repeat_instances,
    remove_spaces,
    count_repeat_words,
    remove_spaces,
]

if __name__ == '__main__':
    runs = [False, True]
    expand_csv_row_size()

    # Generate path
    data_sets = [
        {
            'data_set': 'kaggle',
            'accessor': kaggle_accessor,
            'mutator': kaggle_mutator
        },
        {
            'data_set': 'storm-front-full',
            'accessor': stormfront_accessor,
            'mutator': stormfront_mutator
        },
        {
            'data_set': 'iron_march',
            'accessor': iron_march_accessor,
            'mutator': iron_march_mutator
        },
        {
            'data_set': 'manifesto',
            'accessor': manifesto_accessor,
            'mutator': manifesto_mutator
        },
        {
            'data_set': 'hate_speech_dataset',
            'accessor': hate_speech_dataset_accessor,
            'mutator': hate_speech_dataset_mutator
        },
        {
            'data_set': 'insults',
            'accessor': insults_accessor,
            'mutator': insults_mutator
        },
        {
            'data_set': 'wikipedia_corpus',
            'accessor': wikipedia_corpus_accessor,
            'mutator': wikipedia_corpus_mutator
        }
    ]
    base_directory = make_path('data/')
    source_directory = base_directory / 'datasets'
    dest_directory = base_directory / 'prepared_data'

    check_writable(dest_directory)
    for data_set in data_sets:
        set_name = data_set['data_set']
        check_existence(source_directory / set_name / (set_name + '.csv'))

    # Pre process datasets
    for run_partial_process in runs:
        run_name = 'partial' if run_partial_process else 'pre'
        print('\nRunning', run_name, 'process.')

        processes = partial_processes if run_partial_process else pre_processes

        modified_header = generate_header(processes)

        options = {
            # 'max_documents': 10000,
            'encoding': 'latin-1'
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
