# Function to parse data in place
def parse_data(data, data_formats):
    num_rows = len(data)

    for row_id in range(num_rows):  # For each row in data
        for col_id, formatter in enumerate(data_formats):   # For each column in row of data
            if formatter is not None:
                data[row_id][col_id] = formatter(data[row_id][col_id])


# Debugging function that prints out a two (or more) dimensional array
def print_data(data):
    for row in data:
        print(row)
