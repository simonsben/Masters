from utilities.data_management import move_to_root, make_path

move_to_root()
folder_base = make_path('execution/analysis')

# Generate confusion matrices
with open(folder_base / 'confusion_matrices.py') as fl:
    exec(fl.read())

# Generate feature significance plots
with open(folder_base / 'feature_significance' / 'feature_significance.py') as fl:
    exec(fl.read())
