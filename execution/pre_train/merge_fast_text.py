from utilities.data_management import make_path, check_existence, open_w_pandas, split_sets
from pandas import read_csv
import config

dataset_name = config.dataset

prediction_dir = make_path('data/predictions') / dataset_name
train_path, test_path = prediction_dir / 'train.csv', prediction_dir / 'test.csv'
fast_text_path = prediction_dir / 'fast_text.csv'

check_existence(train_path)
check_existence(test_path)
check_existence(fast_text_path)

train, test = [open_w_pandas(pth) for pth in [train_path, test_path]]
fast_text = read_csv(fast_text_path, index_col=None, header=None).iloc[:, 0].values

tr_length = train.shape[0]
f_train, f_test = fast_text[:tr_length], fast_text[tr_length:]

train['fast_text'] = f_train
test['fast_text'] = f_test

train.to_csv(train_path)
test.to_csv(test_path)
