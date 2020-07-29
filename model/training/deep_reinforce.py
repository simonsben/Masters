from numpy import percentile, min, max, ndarray, argsort, sum
from pandas import DataFrame
from keras.models import Model
from model.layers.realtime_embedding import RealtimeEmbedding
from model.training.rate_limiting import deep_rate_limit
from config import training_verbosity, confidence_increment, batch_size, prediction_threshold


deep_history = None


def save_deep_history(filename):
    """
    Saves a model's training history to a .csv

    :param Path filename: Filename to save the history to
    :param dict history: Dictionary of the training history
    """

    global deep_history
    training_history = DataFrame(deep_history)
    training_history.to_csv(filename)

    if training_verbosity > 0:
        print('Saved training history\n', training_history)


def push_history(epoch_statistics):
    """
    Add epoch's training statistics to aggregated training history

    :param dict epoch_statistics: Dictionary of training statistics from epoch
    """
    if not isinstance(epoch_statistics, dict):
        raise TypeError('Passed statistics must be the dictionary returned by Keras.')

    global deep_history
    if deep_history is None:
        deep_history = epoch_statistics
        return

    for key in deep_history:
        deep_history[key].append(epoch_statistics[key][0])


def print_bits(values):
    print(
        percentile(values, 98),
        percentile(values, 97),
        percentile(values, 95),
        percentile(values, 90),
        percentile(values, 80),
        percentile(values, 70)
    )


def train_deep_learner(model, current_labels, data_source, training_documents=250000, min_confidence=prediction_threshold):
    """
    Performs X rounds of training on deep network to learn from then update the current labels

    :param Model model: Keras model to be trained
    :param ndarray current_labels: Current set of document labels
    :param RealtimeEmbedding data_source: Array of documents with each token enumerated corresponding to word embeddings
    :param int training_documents: Number of documents to train with between rounds [default 250,000]
    :param float min_confidence: Min predicted value for document to *contain intent* [default .985]
    :return model, current labels, new_predictions
    """
    data_source.update_labels(current_labels)

    # Get subset of non uncertain data to use for training
    training_mask = current_labels != .5    # Only use labels that are not *uncertain*
    data_source.set_mask(training_mask)

    training_documents = min((training_documents, sum(training_mask)))
    training_steps = int(training_documents / batch_size)  # Compute number of batches to train each round

    if training_verbosity > 0:
        print('Training for %d steps over %d documents' % (training_steps, training_documents))

    # Train model
    data_source.set_usage_mode(True)
    history = model.fit_generator(data_source, verbose=training_verbosity, steps_per_epoch=training_steps, shuffle=True).history

    push_history(history)

    # Make predictions for all documents
    data_source.set_usage_mode(False)
    predictions = model.predict_generator(data_source, verbose=training_verbosity).reshape(-1)

    # Compute mask of documents with positive and negative intent
    new_positives, new_negatives = deep_rate_limit(predictions, current_labels, min_confidence)

    # Apply confidence modifications to new labels
    current_labels[new_positives] += confidence_increment
    current_labels[new_negatives] -= confidence_increment

    # Bound labels to [0, 1]
    current_labels[current_labels < 0] = 0
    current_labels[current_labels > 1] = 1

    # Print out most intentful contexts for *sanity checking* during training process
    if training_verbosity > 0:
        print('Deep learner changes', sum(new_positives) + sum(new_negatives))

        top_contexts = argsort(predictions)[-30:]
        for index in top_contexts:
            print(predictions[index], data_source.data_source[index])

    return current_labels
