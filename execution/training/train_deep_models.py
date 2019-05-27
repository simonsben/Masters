from utilities.data_management import open_w_pandas, check_existence,  make_path
from model.extraction import load_vectors, vectorize_data
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, Dropout, GlobalMaxPool1D

filename = make_path('../../data/prepared_data/24k-abusive-tweets.csv')
lex_filename = make_path('../../data/lexicons/fast_text/fast_text_min.vec')

check_existence(filename)
check_existence(lex_filename)

dataset = open_w_pandas(filename)

word_vectors, (num_words, vector_dim) = load_vectors(lex_filename)
print('Number of words:', num_words, '\nVector dimensions:', vector_dim)
print(word_vectors)

vectorize_data(dataset, word_vectors)
print(dataset)


embed_layer = Embedding(
    len
)

model = Sequential([

])
