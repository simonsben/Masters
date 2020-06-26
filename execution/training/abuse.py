from model.networks import generate_abuse_network
from utilities.data_management import make_dir, make_path, open_w_pandas, check_existence, \
    get_model_path, load_vector, vector_to_file, split_sets, get_embedding_path
from fasttext import load_model
from model.layers.realtime_embedding import RealtimeEmbedding
from keras.callbacks import EarlyStopping
from config import dataset, max_tokens, training_verbosity, batch_size
from time import time


# Define paths
abuse_weights_path = get_model_path('abuse')
embedding_path = get_embedding_path()
base_path = make_path('data/processed_data/') / dataset / 'analysis'
data_path = make_path('data/prepared_data/abusive_data.csv')
dest_dir = base_path / 'abuse'

# Check for files and make directories
check_existence([embedding_path, data_path])
make_dir(abuse_weights_path.parent)
make_dir(dest_dir)
print('Config complete.')

# Load embeddings and contexts
embedding_model = load_model(str(embedding_path))
labels, documents = open_w_pandas(data_path)[['is_abusive', 'document_content']].sample(frac=1).values.transpose()
labels = labels.astype(bool)
print('Loaded data.')


training_data, testing_data, training_labels, testing_labels = split_sets(documents, labels=labels)

# Generate model
training = RealtimeEmbedding(embedding_model, training_data, training_labels, labels_in_progress=True, uniform_weights=True)
training.set_usage_mode(True)

testing = RealtimeEmbedding(embedding_model, testing_data, testing_labels, labels_in_progress=True, uniform_weights=True)
testing.set_usage_mode(True)


model = generate_abuse_network(max_tokens, embedding_dimension=training.embedding_dimension)
print('Generated model\n', model.summary())

training_steps = int(len(training_data) / batch_size) + 1
validation_steps = int(len(testing_data) / batch_size) + 1

start = time()

stopping_conditions = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
history = model.fit_generator(training, epochs=50, verbose=training_verbosity, callbacks=[stopping_conditions],
                    validation_data=testing, shuffle=True).history

training_time = time() - start
print('Completed training in', training_time, 's')
print('Training history', history)

evaluated_accuracy = model.evaluate_generator(testing, verbose=training_verbosity, steps=validation_steps)
print('Model validation accuracy', evaluated_accuracy)

model.save_weights(str(abuse_weights_path))
print('Completed training and saving abuse model.')

vector_to_file(training.data_source, dest_dir / 'training_data.csv')
vector_to_file(training.labels, dest_dir / 'training_labels.csv')

vector_to_file(testing.data_source, dest_dir / 'testing_data.csv')
vector_to_file(testing.labels, dest_dir / 'testing_labels.csv')
