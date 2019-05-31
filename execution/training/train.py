from os import listdir
from re import compile, search
from utilities.data_management import make_path, move_to_root


file_regex = compile(r'\.py$')
files = listdir('.')
layer_base = make_path('execution/training/')

move_to_root()

exclusions = [
    'train.py',
    # 'deep_models.py',
    'stacked.py'
]

# Train sub-layers
for file in files:
    if search(file_regex, file) is not None and file not in exclusions:
        print('Executing:', file)

        with open(layer_base / file) as fl:
            sub = exec(fl.read())

# Make predictions
predict_files = ['xg_boost.py', 'deep_model.py']
base = make_path('execution/prediction')

for file in predict_files:
    with open(base / file) as fl:
        exec(fl.read())

# Train stacked
with open(base / 'stacked.py') as fl:
    exec(fl.read())

print('\nAll models trained.')
