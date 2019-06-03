from utilities.data_management import move_to_root, make_path

move_to_root()
folder_base = make_path('execution/analysis')

# Generate confusion matrices
print('Generating confusion matrices')
with (folder_base / 'confusion_matrices.py').open() as fl:
    exec(fl.read())

# Generate feature significance plots
print('Generating feature significance figures')
with (folder_base / 'feature_significance' / 'feature_significance.py').open() as fl:
    exec(fl.read())
