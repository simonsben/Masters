from utilities.data_management import make_path, check_existence, check_writable, open_exp_lexicon, \
    check_readable
from os import listdir
from re import compile, match
import config

name_regex = compile(r'[\w\-]+')
dataset_name = config.dataset

source_dir = make_path('data/processed_data') / dataset_name / 'analysis' / 'lexicon_expansion'
dest_dir = make_path('data/prepared_lexicon/')

check_readable(source_dir)
check_writable(dest_dir)

for lexicon_name in listdir(source_dir):
    lexicon = open_exp_lexicon(source_dir / lexicon_name)
    lexicon.to_csv(dest_dir / lexicon_name)

    lex_name = match(name_regex, lexicon_name)[0].replace('_', ' ').capitalize()
    print(lex_name, 'saved')

print('\nAll lexicons assembled.')
