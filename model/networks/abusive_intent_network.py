from model.analysis import compute_abusive_intent
from numpy import vectorize, zeros, asarray


def predict_abusive_intent(documents, embedding_model, abuse_network, intent_network, max_tokens, method='product'):
    """
    Makes abusive intent predictions for a list of pre-processed documents
    :param documents: list or array of pre-processed documents
    :param embedding_model: fastText embedding model
    :param abuse_network: keras network trained to predict abuse
    :param intent_network: keras network trained to predict intent
    :param max_tokens: maximum document length
    :param method: method used to make abusive intent predictions
    :return: tuple of abuse, intent, and abusive-intent predictions
    """
    predictor = abuse_intent_predictor(embedding_model, abuse_network, intent_network, max_tokens)

    abuse_intent_predictions = predictor(documents)
    abuse_predictions, intent_predictions = abuse_intent_predictions.transpose()

    abusive_intent_predictions = compute_abusive_intent(intent_predictions, abuse_predictions, method)

    return abuse_predictions, intent_predictions, abusive_intent_predictions


def abuse_intent_predictor(embedding_model, abuse_network, intent_network, max_tokens):
    """
    Generates a function that computes abusive intent predictions for documents
    :param embedding_model: fastText embedding model
    :param abuse_network: keras network trained to predict abuse
    :param intent_network: keras network trained to predict intent
    :param max_tokens: maximum document length
    :return: function that predicts abuse and intent from pre-processed text
    """
    token_cache = {}
    embedding_dim = embedding_model.get_dimension()

    def make_predictions(document):
        tokens = document.split(' ')
        token_vectors = zeros((max_tokens, embedding_dim))

        for index, token in enumerate(tokens):
            if token not in token_cache:
                token_cache[token] = embedding_model.get_word_vector(token)

            token_vectors[index] = token_cache[token]

        abuse_prediction = abuse_network.predict(token_vectors)
        intent_prediction = intent_network.predict(token_vectors)

        return asarray([abuse_prediction, intent_prediction])

    return vectorize(make_predictions)

