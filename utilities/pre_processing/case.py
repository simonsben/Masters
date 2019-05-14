# Function to count number of upper case characters [and replace with lowercase]
def count_upper(document, replace=True):
    count = sum(1 for ch in document if ch.isupper)

    if replace:
        document = document.lower()

        return count, document
    return count
