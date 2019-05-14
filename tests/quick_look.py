# Some code to give a quick look at some of the data
from utilities.data_management import load_csv, parse_data, print_data
from utilities.pre_processing import count_upper, process_documents

filename = '../datasets/24k-abusive-tweets/24k-abusive-tweets.csv'
data_format = [int, int, int, int, int, int, None]

headers, data = load_csv(filename, has_header=True, limit_rows=20)
parse_data(data, data_format)

print(headers)
print_data(data)

cont_ind = len(data_format) - 1
processes = [count_upper]

