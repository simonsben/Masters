from utilities.data_management import open_w_pandas
from model.extraction import emotions
from time import time

filename = '../data/prepared_data/24k-abusive-tweets.csv'
lex_filename = '../data/prepared_lexicon/nrc_emotion_lexicon.csv'

dataset = open_w_pandas(filename)
lexicon = open_w_pandas(lex_filename)

start = time()
document_matrices, matrix_emotions = emotions(dataset, lexicon)
end = time()
print('Time to execute: ' + str(end - start))

for emotion, matrix in zip(document_matrices, matrix_emotions):
    print(emotion, matrix)

print('doc ', document_matrices[0].to_coo())
