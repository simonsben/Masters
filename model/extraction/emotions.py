from pandas import DataFrame, SparseDataFrame, notnull
from sklearn.feature_extraction.text import CountVectorizer


def get_emotion_indexes(lexicon, emotion):
    """ Gets the indexes associated with a given emotion """
    if type(emotion) is not list:
        return lexicon.index.values[
            notnull(lexicon[emotion])
        ]
    indexes = set()
    for em in emotion:
        indexes.update(lexicon.index.values[
            notnull(lexicon[em])
        ])
    return sorted(list(indexes))


def emotions(dataset, lexicon):
    """
    Takes the emotion lexicon then constructs a full document-term matrix for all terms.

    :param dataset: Dataset of all documents and their content
    :param lexicon: The emotions lexicon AS PROCESSED (see data/prepared_lexicon/nrc_emotion_lexicon.csv)
    :return:
        emotion_matrices: List of emotion document term matrices, each being a DataFrame
        matrix_emotions: List of emotions (provides order of emotion document matrices)
    """

    if type(dataset) is not DataFrame:
        raise TypeError('Dataset must be a (Pandas) Dataframe')

    dictionary = lexicon['word']

    vectorizer = CountVectorizer(vocabulary=dictionary)
    vector_data = vectorizer.transform(dataset['document_content'])

    document_matrix = vector_data.tocsc()

    emotion_matrices, matrix_emotions,  = [], []
    for emotion in lexicon.columns[1:]:
        emotion_dictionary = get_emotion_indexes(lexicon, emotion)

        emotion_matrices.append(
            document_matrix[:, emotion_dictionary].tocsr()
        )

        matrix_emotions.append(emotion)

    return emotion_matrices, matrix_emotions, vectorizer.get_feature_names()
