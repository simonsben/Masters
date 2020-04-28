from numpy import all, sum, where, argmax, percentile, abs, min, max, ndarray
from keras.models import Model
from config import training_verbosity, batch_size


def rescale(values):
    tmp = values + min(values)
    return tmp / max(tmp)


def print_bits(values):
    print(
        percentile(values, 98),
        percentile(values, 97),
        percentile(values, 95),
        percentile(values, 90),
        percentile(values, 80),
        percentile(values, 70)
    )


def train_deep_learner(model, current_labels, enumerated_documents, rounds=3, sub_rounds=3, min_confidence=.985,
                       label_modifier=.4):
    """
    Performs X rounds of training on deep network to reinforce current labels and generate new set
    :param Model model: Keras model to be trained
    :param ndarray current_labels: Current set of document labels
    :param ndarray enumerated_documents: Array of documents with each token enumerated corresponding to word embeddings
    :param int rounds: Number of reinforcement-training rounds [default 3]
    :param int sub_rounds: Number of epochs in each reinforcement-training round [default 3]
    :param float min_confidence: Min predicted value for document to *contain intent* [default .985]
    :param float label_modifier: Modifier applied to current labels [default .4]
    :return model, current labels, new_predictions
    """
    positive_threshold = min_confidence
    negative_threshold = (1 - min_confidence)

    current_labels = current_labels.copy()
    predictions = None

    for round_number in range(rounds):
        training_mask = current_labels != .5    # Only use labels that are not *uncertain*

        # Get subset of non uncertain data to use for training
        training_data = enumerated_documents[training_mask]
        training_labels = current_labels[training_mask]

        # Train model
        model.fit(training_data, training_labels, batch_size=batch_size, epochs=sub_rounds, verbose=training_verbosity)

        # Make predictions for all documents
        predictions = model.predict(enumerated_documents, verbose=training_verbosity, batch_size=batch_size).reshape(-1)

        # Scale and shift predictions to [0, 1]
        scaled_predictions = rescale(predictions)
        print_bits(scaled_predictions)

        # Compute mask of documents with positive and negative intent
        new_positives = scaled_predictions > positive_threshold
        new_negatives = scaled_predictions < negative_threshold

        # Get indexes of documents with positive and negative intent
        [pos_indices] = where(new_positives)
        [neg_indices] = where(new_negatives)

        # Apply confidence modifications to new labels
        current_labels[pos_indices] += label_modifier
        current_labels[neg_indices] -= label_modifier

        # Restrict label value range to [0, 1]
        current_labels[current_labels < 0] = 0
        current_labels[current_labels > 1] = 1

        print(pos_indices.shape[0] + neg_indices.shape[0], 'classified in deep training round', round_number + 1)

    return model, current_labels, predictions
