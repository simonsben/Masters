from re import compile
from utilities.pre_processing.basic_statistics import partial_clean
from utilities.data_management.io import prepare_csv_writer

special_group = compile(r'((\\.)|[^a-zA-Z0-9])\1+')
bracket_info = compile(r'\([^)]+\)')
quote = compile(r'".+"')


def basic_cleaning(content, _b, _c, _d):
    content = special_group.sub(' ', content)   # Remove wikipedia formatting characters
    content = quote.sub('', content)
    content = bracket_info.sub(' ', content)     # Remove bracketed information that *disrupts* the sentence
    content = partial_clean.sub(' ', content)    # Apply partial cleaning to the content

    return content


def make_corpus(in_f, out_f, max_files=None):
    """
    Convert Wikipedia xml dump file to text corpus
    From: https://www.kdnuggets.com/2017/11/building-wikipedia-text-corpus-nlp.html
    """

    writer, fl = prepare_csv_writer(out_f, ['', 'document_content'])
    wiki = WikiCorpus(in_f, lemmatize=False, dictionary=set(), tokenizer_func=basic_cleaning)

    i = 0
    for document in wiki.get_texts():
        writer.writerow((i, document))

        i += 1
        if max_files is not None and i >= max_files:
            print('File limit hit.')
            break
        elif i % 10000 == 0:
            print('Processed', i, 'articles')


if __name__ == '__main__':
    from utilities.data_management import make_path
    from gensim.corpora import WikiCorpus

    base = make_path('data/datasets/wikipedia_corpus/')
    source_file = base / 'wikipedia_corpus.xml.bz2'
    output_file = base / 'wikipedia_corpus.csv'

    make_corpus(source_file, output_file, max_files=5000000)
