from model.high_order.job_runner import job_runner
from utilities.data_management import move_to_root, make_path, load_execution_params
from model.expansion.intent_seed import worker_init, identify_basic_intent
from model.extraction import context_breaks


def process(document):
    if not isinstance(document[0], str):
        return []
    return [[identify_basic_intent(context), context] for context in context_breaks.split(document[0])]


def data_modifier(document, making_header=False):
    if making_header:
        return ['has_intent', 'context']
    return [document[-1]]


if __name__ == '__main__':
    move_to_root()

    params = load_execution_params()
    dataset_name = params['dataset']

    source = make_path('data/prepared_data/') / (dataset_name + '_partial.csv')
    dest = make_path('data/processed_data/') / dataset_name / 'analysis' / 'intent' / 'alt_intent_mask.csv'

    runner = job_runner(process, source, dest, worker_init=worker_init, chunk_size=500)

    runner.process_documents(data_modifier, max_documents=25, list_return=True)
