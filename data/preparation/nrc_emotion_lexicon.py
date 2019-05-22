from pandas import read_csv
from utilities.data_management import check_readable, check_writable

source_filename = '../lexicons/NRC-Sentiment-Emotion-Lexicons/' \
                  'NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
dest_filename = '../prepared_lexicon/nrc_emotion_lexicon.csv'

check_readable(source_filename)
check_writable(dest_filename)

header = [
    'word',
    'emotion',
    'flag'
]

# Read raw lexicon
raw_lexicon = read_csv(source_filename, delimiter='\t', names=header)
raw_lexicon = raw_lexicon[raw_lexicon['flag'] != 0].reindex()

# Create a sparse matrix with each lexicon as a column
lexicon = raw_lexicon.pivot(index='word', columns='emotion', values='flag').to_sparse()

# Save as csv (not as efficient as using sparse format, but its simple)
lexicon.to_csv(dest_filename)
