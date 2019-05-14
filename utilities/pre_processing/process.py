

# Function to process all documents with processes given
# Processes are assumed to be of the form (document, index) -> (value, modified_document)
def process_documents(documents, processes, cont_ind=-1):
    use_index = cont_ind != -1

    for ind, doc in enumerate(documents):
        target = doc if not use_index else doc[cont_ind]
        values = []

        for process in processes:
            value, modified = process(target, cont_ind=cont_ind)
            values.append(value)

            if use_index:
                documents[ind][cont_ind] = modified
            else:
                documents[ind] = modified

    return values
