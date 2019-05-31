from utilities.data_management import move_to_root, make_path

move_to_root()
base = make_path('execution/')

# Train models
with open(base / 'training' / 'train.py') as fl:
    exec(fl.read())

# Run final prediction
with open(base / 'prediction' / 'stacked.py') as fl:
    exec(fl.read())
