from utilities.data_management import make_path

base = make_path('execution/')

# Train models
with open(base / 'training' / 'train.py') as fl:
    exec(fl.read())

# Run final prediction
with open(base / 'prediction' / 'stacked.py') as fl:
    exec(fl.read())
