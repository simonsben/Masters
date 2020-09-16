from utilities.data_management import make_path, open_w_pandas

base = make_path('data/prepared_data/')
source_name = 'storm-front-full'
destination_name = 'storm-front'

variants = ['', '_partial']
sample_size = 250000
print('Prepared params, starting selection')

sample_indexes = None

for variant in variants:
    source_path = base / (source_name + variant + '.csv')
    destination_path = base / (destination_name + variant + '.csv.gz')

    source = open_w_pandas(source_path)
    print('Loaded data variant', variant)

    # Define the same subset for both variants
    if sample_indexes is None:
        sample = source.sample(n=sample_size)
        sample_indexes = sample.index.values
    else:
        sample = source.iloc[sample_indexes]

    sample.to_csv(destination_path)
    print('Finished variant', variant)
print('All writes complete.')
