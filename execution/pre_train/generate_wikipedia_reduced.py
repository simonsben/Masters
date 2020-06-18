from utilities.data_management import make_path, open_w_pandas, check_existence

base = make_path('data/prepared_data/')
original_path = base / 'wikipedia_corpus.csv'
check_existence(original_path)

num_documents = 50000
modifiers = ['', '_partial']

print('Config complete.')

sample_indexes = None

for modifier in modifiers:
    data = open_w_pandas(base / ('wikipedia_corpus' + modifier + '.csv'))
    print('Loaded ' + modifier + ' data.')

    if sample_indexes is None:
        data = data.sample(n=num_documents)
        sample_indexes = data.index.values
    else:
        data = data.iloc[sample_indexes]
    print('Sampled data.')

    data.to_csv(base / ('wikipedia_corpus_reduced' + modifier + '.csv'))
    print('Saved data.')
