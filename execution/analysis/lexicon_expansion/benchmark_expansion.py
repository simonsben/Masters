from utilities.data_management import make_path, move_to_root, check_existence, check_readable, open_w_pandas, \
    open_exp_lexicon, make_dir
from os import listdir
from sklearn.feature_extraction.text import CountVectorizer
from model.training import train_xg_boost
from numpy import sum
from utilities.plotting import scatter_plot
from re import compile, match
from matplotlib.pyplot import show, savefig
import config

name_regex = compile(r'[\w\-]+')

move_to_root(4)

# Load execution parameters
dataset_name = config.dataset

# Define file paths
expanded_dir = make_path('data/processed_data') / dataset_name / 'analysis' / 'lexicon_expansion'
dataset_path = make_path('data/prepared_data/') / (dataset_name + '.csv')
fig_path = make_path('figures/') / dataset_name / 'analysis' / 'lexicon_expansion'

# Check for files/readable/writeable
check_readable(expanded_dir)
check_existence(dataset_path)
make_dir(fig_path)

# Load dataset
dataset = open_w_pandas(dataset_path)
documents = dataset['document_content'].values
is_abusive = dataset['is_abusive'].values

expansion_accuracies = []

# For each expanded lexicon
for file_name in listdir(expanded_dir):
    lexicon_name = match(name_regex, file_name)[0].replace('_', ' ').capitalize()
    print('Starting', lexicon_name)

    # Load lexicon
    expanded = open_exp_lexicon(expanded_dir / file_name, True)
    lexicon_accuracies = []

    # Slice into original and added terms
    original_terms = expanded[0]
    new_terms = expanded[1:]

    # Initialize list of level sets and calculate number of levels
    expanded_lexicon = set(original_terms)
    num_syms = max([len(syms) for syms in new_terms])

    # For each level
    for level in range(num_syms):
        print('Starting level set', level)

        # Add level terms to lexicon
        for syms in new_terms:
            if len(syms) > level:
                expanded_lexicon.add(syms[level])

        # Initialize vectorizer and generate document_matrix
        vectorizer = CountVectorizer(vocabulary=list(expanded_lexicon))
        document_matrix = vectorizer.transform(documents)

        # Train classifier and make predictions on test set
        classifier, (_, (te_docs, te_abus)) = train_xg_boost(document_matrix, is_abusive, return_data=True)
        preds = classifier.predict(te_docs)

        # Calculate level accuracy
        accuracy = sum(preds == te_abus) / te_docs.shape[0]

        lexicon_accuracies.append(accuracy)
    expansion_accuracies.append(lexicon_accuracies)

    # Plot level accuracies and save
    ax = scatter_plot(
        (list(range(len(lexicon_accuracies))), lexicon_accuracies),
        lexicon_name + ' expansion accuracy',
    )
    ax.set_xlabel('New terms added per original term')
    ax.set_ylabel('Accuracy on test set')
    savefig(fig_path / (match(name_regex, file_name)[0] + '.png'))


print('Accuracies computed')
show()
