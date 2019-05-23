from utilities.data_management import open_w_pandas
from model.extraction import load_vectors, vectorize_data
from time import time

filename = '../data/prepared_data/24k-abusive-tweets.csv'
lex_filename = '../data/lexicons/fast_text/fast_text.vec'

# dataset = open_w_pandas(filename)
word_vectors, (num_words, vector_dim) = load_vectors(lex_filename)

# start = time()
# vectorize_data(dataset, word_vectors)
# end = time()
# print('Time to execute: ' + str(end - start))

# print(dataset)

print('Number of words: ', num_words, '\nVector dimensions: ', vector_dim)
print(word_vectors)
