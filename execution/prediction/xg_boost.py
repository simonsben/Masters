from os import listdir, mkdir
from utilities.data_management import move_to_root, make_path, check_readable, check_writable, load_execution_params
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame
from scipy.sparse import load_npz

move_to_root()

dataset = load_execution_params()['dataset']
model_base_path = make_path('data/models/') / dataset
data_base_path = make_path('data/processed_data/') / dataset
predictions_path = make_path('data/predictions') / dataset

if not predictions_path.exists():
    mkdir(predictions_path)

check_readable(model_base_path)
check_readable(data_base_path)
check_writable(predictions_path)

test_predictions = DataFrame()
train_predictions = DataFrame()

models = []
sub_dirs = listdir(model_base_path)
for sub_dir in sub_dirs:
    files = listdir(model_base_path / sub_dir)

    for file in files:
        file_path = make_path(file)
        if file_path.suffix == '.bin':
            print('Loading and executing', file_path.stem)

            model = XGBClassifier(objective='binary:logistic', n_estimators=600, silent=True)
            model.load_model(str(model_base_path / sub_dir / file))
            model._le = LabelEncoder().fit([0, 1])

            data_base = data_base_path / sub_dir
            train_set = load_npz(data_base / (file_path.stem + '_train.npz'))
            test_set = load_npz(data_base / (file_path.stem + '_test.npz'))

            train_predictions[file_path.stem] = model.predict_proba(train_set)[:, 1]
            test_predictions[file_path.stem] = model.predict_proba(test_set)[:, 1]

print(test_predictions.describe())
print(train_predictions.describe())

test_predictions.to_csv(predictions_path / 'test.csv')
train_predictions.to_csv(predictions_path / 'train.csv')
