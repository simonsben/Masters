from model.networks import generate_abuse_network
from utilities.data_management import make_dir, make_path, open_w_pandas, check_existence, \
    get_model_path, load_vector, vector_to_file, split_sets
from fasttext import load_model
from model.layers.realtime_embedding import RealtimeEmbedding
from keras.callbacks import EarlyStopping
from config import dataset, max_tokens, training_verbosity, batch_size


# Define paths
abuse_weights_path = get_model_path('intent')
embedding_path = make_path('data/models/') / dataset / 'derived' / (dataset + '.bin')
base_path = make_path('data/processed_data/') / dataset / 'analysis'
data_path = make_path('data/prepared_data/abusive_data.csv')

# Check for files and make directories
check_existence([embedding_path, data_path])
make_dir(abuse_weights_path.parent)
print('Config complete.')

# Load embeddings and contexts
embedding_model = load_model(str(embedding_path))
labels, documents = open_w_pandas(data_path)[['is_abusive', 'document_content']].sample(frac=1).values.transpose()
labels = labels.astype(bool)
print('Loaded data.')


training_data, testing_data, training_labels, testing_labels = split_sets(documents, labels=labels)

# Generate model
training = RealtimeEmbedding(embedding_model, training_data, training_labels, mark_initial_labels=True, uniform_weights=True)
training.set_usage_mode(True)

testing = RealtimeEmbedding(embedding_model, testing_data, testing_labels, mark_initial_labels=True, uniform_weights=True)
testing.set_usage_mode(True)


model = generate_abuse_network(max_tokens, embedding_dimension=training.embedding_dimension)
print('Generated model\n', model.summary())

training_steps = int(len(training_data) / batch_size) + 1
validation_steps = int(len(testing_data) / batch_size) + 1

stopping_conditions = EarlyStopping(monitor='val_loss', patience=2, verbose=1, restore_best_weights=True)
model.fit_generator(training, epochs=50, verbose=training_verbosity, callbacks=[stopping_conditions],
                    validation_data=testing, shuffle=True)


evaluated_accuracy = model.evaluate_generator(testing, verbose=training_verbosity, steps=validation_steps)
print('Model validation accuracy', evaluated_accuracy)

model.save_weights(str(abuse_weights_path))
