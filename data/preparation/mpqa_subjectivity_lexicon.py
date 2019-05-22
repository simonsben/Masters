from utilities.data_management import make_path, check_readable, check_writable
from pandas import read_csv
from re import compile, search

# Define filenames and paths
filename = 'mpqa_subjectivity_lexicon'
source_filename = '../lexicons/' + filename + '/' + filename + '.tff'
dest_filename = '../prepared_lexicon/' + filename + '.csv'
source_path = make_path(source_filename)
dest_path = make_path(dest_filename)
check_readable(source_path)
check_writable(dest_path)

# Define constants
header = [
    'is_strong',
    'length',
    'word',
    'pos',
    'is_stemmed',
    'polarity'
]
value_regex = compile(r'(?<==)\w')
word_regex = compile(r'(?<=\w=)\w+')
pos_regex = compile(r'(?<=\w=)\w+')

# Read lexicon
lexicon = read_csv(source_path, delimiter=' ', names=header, usecols=[header[0]] + header[2:5])

# Clean lexicon
lexicon['is_strong'] = lexicon['is_strong'].apply(lambda doc: search(value_regex, doc).group(0) == 's')
lexicon['is_stemmed'] = lexicon['is_stemmed'].apply(lambda doc: search(value_regex, doc).group(0) == 'y')
lexicon['word'] = lexicon['word'].apply(lambda doc: search(word_regex, doc).group(0))
lexicon['pos'] = lexicon['pos'].apply(lambda doc: search(pos_regex, doc).group(0))

# Save lexicon
lexicon.to_csv(dest_path)
