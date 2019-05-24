from utilities.data_management import open_w_pandas
from model.extraction import hurtlex
from model.training import train_xg_boost
from time import time

filename = '../data/prepared_data/24k-abusive-tweets.csv'
lex_filename = '../data/prepared_lexicon/hurtlex.csv'

dataset = open_w_pandas(filename)
lexicon = open_w_pandas(lex_filename)

start = time()
document_matrix = hurtlex(dataset, lexicon)
end = time()

print('Time to execute: ' + str(end - start))
print(document_matrix)

model, [test_data, test_labels] = train_xg_boost(document_matrix, dataset['is_abusive'], return_test=True)

print(model)
print(model.feature_importances_)

for word, val in zip(document_matrix.columns, model.feature_importances_):
    if val != 0:
        print(word, val)


predictions = model.predict(test_data)
print('predictions ', predictions)
