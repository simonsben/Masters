from model.high_order.job_runner import job_runner
from utilities.data_management import make_path, expand_csv_row_size
from model.expansion.intent_seed import worker_init, identify_basic_intent
from utilities.pre_processing import split_pattern
import config


def process(document):
    if not isinstance(document[0], str):
        return []
    return [[identify_basic_intent(context), context] for context in split_pattern.split(document[0])]


def data_modifier(document, making_header=False):
    if making_header:
        return ['has_intent', 'context']
    return [document[-1]]


if __name__ == '__main__':
    expand_csv_row_size()
    dataset_name = config.dataset

    source = make_path('data/prepared_data/') / (dataset_name + '_partial.csv')
    dest = make_path('data/processed_data/') / dataset_name / 'analysis' / 'intent' / 'alt_intent_mask.csv'

    runner = job_runner(process, source, dest, worker_init=worker_init, chunk_size=50000, worker_lifespan=50000)

    runner.process_documents(data_modifier, list_return=True)
