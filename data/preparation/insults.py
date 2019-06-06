from utilities.data_management import open_w_pandas, make_path
from pandas import concat

base_path = make_path('../datasets/insults/')

test_data = open_w_pandas(base_path / 'test_with_solutions.csv', index_col=None).drop(columns='Usage')
train_data = open_w_pandas(base_path / 'train.csv', index_col=None)
print('Data loaded.')

dataset = concat([test_data, train_data]).drop(columns='Date').sample(n=3947)
print('Insults combined, saving..')

dataset.to_csv(base_path / 'insults.csv')
