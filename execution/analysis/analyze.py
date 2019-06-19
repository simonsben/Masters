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

# Generate sub-layer histograms
print('Generating sub-layer histograms')
with (folder_base / 'prediction_hist.py').open() as fl:
    exec(fl.read())


# Generating false prediction summaries
print('Generating false prediction summaries')
with (folder_base / 'false_predictions.py').open() as fl:
    exec(fl.read())
