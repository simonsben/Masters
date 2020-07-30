from config import training_verbosity, confidence_increment
from model.layers.realtime_embedding import compute_sample_weights
from xgboost import XGBClassifier
from scipy.sparse import csr_matrix
from utilities.analysis import get_feature_values
from utilities.data_management import match_feature_weights
from numpy import where

midpoint = 0.5


def reinforce_xgboost(model, context_matrix, labels, initial_labels, features=None, min_confidence=.985):
    """
    Performs *reinforcement* cycle using the xgboost tree learner

    :param XGBClassifier model: XGBoost model used
    :param csr_matrix context_matrix: Sparse sequence-context matrix of contexts
    :param ndarray labels: Array of current training labels
    :return ndarray: Updates labels
    """
    # Copy labels to ensure function does not cause *side effects*
    labels = labels.copy()

    # Determine contexts to use for training and compute their weights
    working_mask = labels != .5
    boolean_labels = labels[working_mask] >= midpoint
    weights = compute_sample_weights(labels[working_mask], midpoint)

    [validation_indexes] = where(initial_labels != 0.5)
    validation = [(context_matrix[index], initial_labels[index]) for index in validation_indexes]
    # validation = (context_matrix[validation_mask], initial_labels[validation_mask])

    # Train model with working contexts and make predictions about all documents
    model.fit(context_matrix[working_mask], boolean_labels, weights, verbose=training_verbosity,
              early_stopping_rounds=1, eval_set=validation)
    print('Trained')
    predictions = model.predict_proba(context_matrix)
    print('Predicted')

    # Compute thresholds to accept predictions
    positive_threshold = min_confidence
    negative_threshold = (1 - min_confidence)

    # Identify to-be-labelled contexts
    new_positives = predictions > positive_threshold
    new_negatives = predictions < negative_threshold

    # Apply confidence modifications to new labels
    labels[new_positives] += confidence_increment
    labels[new_negatives] -= confidence_increment

    # Bound labels to [0, 1]
    labels[labels < 0] = 0
    labels[labels > 1] = 1

    if training_verbosity > 0 and features is not None:
        feature_weights = match_feature_weights(features, get_feature_values(model))
        for feature in feature_weights[:20]:
            print(feature)

    return labels
