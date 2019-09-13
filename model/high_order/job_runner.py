from utilities.data_management import in_parent_dir, load_execution_params, move_to_root, prepare_csv_reader, \
    prepare_csv_writer
from multiprocessing import Pool
from pathlib import Path
from time import time
from functools import partial


def generate_data_modifier(header_list, data_header):
    """ Generates a function to select the desired columns from the source data """
    indexes = [data_header[header] for header in header_list]
    max_ind = max(indexes)

    def data_modifier(document):
        # If document does not contain enough fields, return list of None values
        if len(document) < max_ind:
            return [None] * len(indexes)
        return [document[index] for index in indexes]
    return data_modifier


def get_path(path):
    """ Function that ensures that the path given is either a string or Path (and converts to Path) """
    if isinstance(path, (Path, str)):
        return Path(path)
    raise TypeError('Path given must be of type string or Path')


def check_processes(processes):
    """ Checks passed processes to ensure they are callable """
    if callable(processes):
        return [processes]
    elif isinstance(processes, list):
        for ind, p in enumerate(processes):
            if not callable(p):
                raise ValueError('Process ' + str(ind + 1) + ' provided is not a function')
        return processes


def processor(document, processes):
    """ Function that applies multiple process functions to each document """
    for process in processes:
        document = process(document)
    return document


class job_runner:
    """ Class for long-running processes with large datasets """
    def __init__(self, processes, source_path, dest_path, n_threads=None, dataset_name=None, worker_init=None,
                 chunk_size=10000, worker_lifespan=None):
        self.get_default_params()   # Load defaults

        self.processor = check_processes(processes) # generate_processor(check_processes(processes))
        self.worker_init = worker_init
        self.chunk_size = chunk_size
        self.worker_lifespan = worker_lifespan
        self.workers = None

        # If alternate parameter value is provided, take it
        if n_threads is not None: self.n_threads = n_threads
        if dataset_name is not None: self.dataset_name = dataset_name

        self.source_path = get_path(source_path)
        self.dest_path = get_path(dest_path)
        self.csv_reader = self.source_file = self.data_header = self.data_modifier = self.csv_writer = self.dest_file \
            = None
        self.data_ready = self.workers_ready = False

    def get_default_params(self):
        """ Loads default parameters from config file """
        if not in_parent_dir():
            move_to_root()

        params = load_execution_params()
        self.n_threads = params['n_threads']
        self.dataset_name = params['dataset']

    def start_workers(self):
        """ Starts worker threads """
        self.workers = Pool(self.n_threads, initializer=self.worker_init, maxtasksperchild=self.worker_lifespan)
        self.workers_ready = True
        print('Workers ready.')

    def shutdown_workers(self):
        """ Stops all worker processes """
        if self.workers_ready:
            self.workers.close()
            self.workers.join()
            print('Workers killed.')

    def prepare_data(self, data_modifier=None):
        """ Opens buffered file-stream with source and destination files """
        self.csv_reader, self.source_file, data_header = prepare_csv_reader(self.source_path)
        self.data_header = {header: index for index, header in enumerate(data_header)}

        if data_modifier is None:
            self.data_modifier = lambda x: x
        elif isinstance(data_modifier, list):
            for item in data_modifier:
                if not isinstance(item, str) and item in self.data_header:
                    raise ValueError('Illegal data modifier, if list of strings is passed, items must match data '
                                     'headers')
            self.data_modifier = generate_data_modifier(data_modifier, self.data_header)

        dest_header = [''] + self.data_modifier(data_header)
        self.csv_writer, self.dest_file = prepare_csv_writer(self.dest_path, dest_header)

        self.data_ready = True
        print('Data prepared.')

    def finalize_data(self):
        """ Closes file-streams """
        self.source_file.close()
        self.dest_file.close()
        print('Document finalized.')

    def process_documents(self, data_modifier=None, single_job=True, max_documents=None):
        """ Processes documents """
        # While there is still data to process
        if not self.data_ready:
            self.prepare_data(data_modifier)
        if not self.workers_ready:
            self.start_workers()

        print('Starting document processing')
        eof_reached = False
        documents_processed = chunk_number = 0

        while True:
            chunk_number += 1

            data_start = time()
            data, index = [], 0

            # Enable max_documents functionality
            limit = self.chunk_size
            if max_documents is not None and self.chunk_size + documents_processed > max_documents:
                limit = max_documents

            # Pull documents
            for index in range(limit):
                try:
                    document = next(self.csv_reader)
                    data.append(self.data_modifier(document))
                except StopIteration:
                    eof_reached = True
                    break

            documents_processed += index + 1
            data_time = time() - data_start

            # Process documents
            process_start = time()
            data = self.workers.map(partial(processor, processes=self.processor), data)
            process_time = time() - process_start

            # If it takes more time to load the data then run the process, get larger chunks at a time
            if process_time < data_time:
                self.chunk_size *= 2
                print('Process time less than data load time, doubling chunk size to', self.chunk_size)

            self.csv_writer.writerows(enumerate(data, start=documents_processed - limit))
            print('Finished chunk', chunk_number)

            # If there are no more documents or there is a max document restriction (that is met)
            if eof_reached or (max_documents is not None and max_documents <= documents_processed):
                break

        # If this is the only job to be run, close files and kill workers
        if single_job:
            self.shutdown_workers()
            self.finalize_data()
