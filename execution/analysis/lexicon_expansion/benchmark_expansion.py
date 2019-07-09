from utilities.data_management import make_path, move_to_root, check_existence, load_execution_params, check_readable, \
    open_w_pandas, open_exp_lexicon, make_dir
from os import listdir
from sklearn.feature_extraction.text import CountVectorizer
from model.training import train_xg_boost
from numpy import sum
from utilities.plotting import scatter_plot
from re import compile, match
from matplotlib.pyplot import show, savefig

name_regex = compile(r'[\w\-]+')

move_to_root(4)

params = load_execution_params()
dataset_name = params['dataset']

expanded_dir = make_path('data/processed_data') / dataset_name / 'analysis' / 'lexicon_expansion'
dataset_path = make_path('data/prepared_data/') / (dataset_name + '.csv')
fig_path = make_path('figures/') / dataset_name / 'analysis' / 'lexicon_expansion'

check_readable(expanded_dir)
check_existence(dataset_path)
make_dir(fig_path)

dataset = open_w_pandas(dataset_path)
documents = dataset['document_content'].values
is_abusive = dataset['is_abusive'].values

expansion_accuracies = []

for file_name in listdir(expanded_dir):
    lexicon_name = match(name_regex, file_name)[0].replace('_', ' ').capitalize()
    print('Starting', lexicon_name)

    expanded = open_exp_lexicon(expanded_dir / file_name, True)
    lexicon_accuracies = []

    original_terms = expanded[0]
    new_terms = expanded[1:]

    expanded_lexicons = [set(original_terms)]
    index = len(expanded_lexicons[0])

    num_syms = max([len(syms) for syms in new_terms])

    for ind in range(num_syms):
        level_set = expanded_lexicons[ind].copy()
        expanded_lexicons.append(level_set)

        for syms in new_terms:
            if len(syms) > ind:
                level_set.add(syms[ind])

    print('Level sets generated, benchmarking')

    for level, level_set in enumerate(expanded_lexicons):
        print('Starting level set', level)
        vectorizer = CountVectorizer(vocabulary=list(level_set))
        document_matrix = vectorizer.transform(documents)

        classifier, (_, (te_docs, te_abus)) = train_xg_boost(document_matrix, is_abusive, return_data=True)
        preds = classifier.predict(te_docs)
        accuracy = sum(preds == te_abus) / te_docs.shape[0]

        lexicon_accuracies.append(accuracy)
    expansion_accuracies.append(lexicon_accuracies)

    ax = scatter_plot(
        (list(range(len(lexicon_accuracies))), lexicon_accuracies),
        lexicon_name + ' expansion accuracy',
    )
    ax.set_xlabel('New terms added per original term')
    ax.set_ylabel('Accuracy on test set')
    savefig(fig_path / (match(name_regex, file_name)[0] + '.png'))

print('Accuracies computed')

show()
