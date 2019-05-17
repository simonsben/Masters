# Run a test of the vectorizer
from model.extraction import vectorize_source

path = '../data/prepared_data/24k-abusive-tweets.csv'

doc_matrix = vectorize_source(path, use_words=False)
print(doc_matrix)
