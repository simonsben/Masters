from utilities.data_management import make_path, make_dir, check_existence, load_vector, open_w_pandas, vector_to_file
from model.networks import predict_abusive_intent
from utilities.plotting import confusion_matrix, show
from config import dataset

midpoint = .5

base = make_path('data/processed_data') / dataset / 'analysis' / 'abuse'
data_path = base / 'testing_data.csv'
label_path = base / 'testing_labels.csv'
prediction_path = base / 'testing_predictions.csv'
base_path = make_path('data/processed_data/abusive_data/analysis/intent_abuse')
figure_base = make_path('figures') / dataset / 'analysis'

check_existence([data_path])
make_dir(figure_base)
print('Complete config.')

documents = load_vector(data_path)
labels = load_vector(label_path).astype(bool)

if prediction_path.exists():
    print('Loading data')
    predictions = load_vector(prediction_path) > midpoint
else:
    print('Making abuse predictions')
    predictions, _, _ = predict_abusive_intent(documents)

    vector_to_file(predictions, prediction_path)
    predictions = predictions > midpoint

confusion_matrix(predictions, labels, 'Confusion matrix of abuse model predictions',
                 figure_base / 'abuse_confusion_matrix.png')

show()
