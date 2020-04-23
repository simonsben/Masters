from utilities.data_management import make_path
from pandas import read_csv
from numpy.random import choice

base = make_path('data/prepared_data/')
source_name = 'storm-front-full'
destination_name = 'storm-front'

variants = ['', '_partial']
num_lines = 250000
print('Prepared params, starting selection')

subset = None

for variant in variants:
    source_path = base / (source_name + variant + '.csv')
    destination_path = base / (destination_name + variant + '.csv.gz')

    source = read_csv(source_path, index_col=0)
    print('Loaded data variant', variant)

    # Define the same subset for both variants
    if subset is None:
        subset = choice(source.shape[0], size=num_lines, replace=False)
        print('Defined subset')

    sample = source.iloc[subset]
    sample.to_csv(destination_path)

    print('Finished variant', variant)
print('All writes complete.')
