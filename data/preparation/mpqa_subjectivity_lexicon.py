from utilities.data_management import make_path, check_readable, check_writable
from pandas import read_csv
from numpy import where
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
lexicon = read_csv(source_path, delimiter=' ', names=header, usecols=[header[0]] + header[2:3])

# Clean lexicon
lexicon['is_strong'] = lexicon['is_strong'].apply(lambda doc: search(value_regex, doc).group(0) == 's')
lexicon['word'] = lexicon['word'].apply(lambda doc: search(word_regex, doc).group(0))
# lexicon['pos'] = lexicon['pos'].apply(lambda doc: search(pos_regex, doc).group(0))


def get_strong(data, ind):
    """ Returns the highest weight of the word (i.e. strong if ever strong, else weak) """
    is_strong = data['is_strong'][ind] or False
    if is_strong:
        return True

    ref = data['word'][ind]
    ind += 1
    while data['word'][ind] == ref:
        is_strong = is_strong or data['is_strong'][ind]
        ind += 1

    return is_strong


duplicated = lexicon.duplicated(subset='word', keep='last')
for ind, val in enumerate(duplicated):
    if val is True:
        lexicon.loc[ind, 'is_strong'] = get_strong(lexicon, ind)

lexicon.drop_duplicates(subset='word', inplace=True)
lexicon['score'] = where(lexicon['is_strong'], 2, 1)
lexicon.drop(labels='is_strong', axis=1, inplace=True)

# Save lexicon
lexicon.to_csv(dest_path)
