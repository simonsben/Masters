from utilities.data_management import open_w_pandas
from model.extraction import subjectivity
from time import time

filename = '../data/prepared_data/24k-abusive-tweets.csv'
lex_filename = '../data/prepared_lexicon/mpqa_subjectivity_lexicon.csv'

dataset = open_w_pandas(filename)
lexicon = open_w_pandas(lex_filename)

start = time()
document_matrix = subjectivity(dataset, lexicon)
end = time()
print('Time to execute: ' + str(end - start))
print(document_matrix)
