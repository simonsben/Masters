def parse_data(data, data_formats):
    """ Function to parse 2D list data """
    num_rows = len(data)

    for row_id in range(num_rows):  # For each row in data
        for col_id, formatter in enumerate(data_formats):   # For each column in row of data
            if formatter is not None:
                data[row_id][col_id] = formatter(data[row_id][col_id])


def print_data(data):
    """ Debugging function to print 2D list """
    for row in data:
        print(row)
