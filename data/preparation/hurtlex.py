from pandas import read_csv
from utilities.data_management import check_readable, check_writable
from unidecode import unidecode

source_filename = '../lexicons/hurtlex/hurtlex_EN_conservative.tsv'
dest_filename = '../prepared_lexicon/hurtlex.csv'

check_readable(source_filename)
check_writable(dest_filename)

lexicon = read_csv(source_filename, delimiter='\t',
                   names=['category', 'macro-category', 'lemma'], usecols=['lemma'])
lexicon.rename(index=str, columns={'lemma': 'word'}, inplace=True)

lexicon['word'] = lexicon['word'].apply(unidecode)
lexicon.drop_duplicates(subset='word', inplace=True)
lexicon.to_csv(dest_filename)
