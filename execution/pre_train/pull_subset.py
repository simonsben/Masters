from utilities.data_management import make_path, move_to_root
from pandas import read_csv
from numpy.random import choice

move_to_root()

base = make_path('data/prepared_data/')
source_name = 'storm-front-full'
dest_name = 'storm-front'

variants = ['', '_partial']
num_lines = 250000
print('Prepared params, starting selection')

subset = None

for variant in variants:
    source_path = base / (source_name + variant + '.csv')
    dest_path = base / (dest_name + variant + '.csv.gz')

    source = read_csv(source_path, index_col=0)
    print('Loaded data')

    # Define the same subset for both variants
    if subset is None:
        subset = choice(source.shape[0], size=num_lines, replace=False)
        print('Defined subset')

    sample = source.iloc[subset]
    sample.to_csv(dest_path)

    print('Finished', variant)
print('All writes complete.')
