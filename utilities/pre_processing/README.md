# Pre-processing

The purpose of this README is to describe implementation and style specifics, for *methodology* related notes see the [other README](../../notes/pre_processing.md).

## Code Structure

The pre-processor is comprised of four main parts, 

* Processor
* Filters
* Accessors
* I/O interaction

The top-level is the processor, which takes all of the following components and executes the pre-processing.
For syntax specifics see the docstring within the [processor](process.py) function.

The filters are required by the processor as a List.
The general flow of each filter is `(document) -> (statistic, modified_document)`.
Additionally, every filter must have an optional secondary parameter, `get_header=False`, which provides the filter name.
However, if a filter can return `None` for its statistic and name, if no statistic is related.

The accessors provide dataset specific data routing for the processor.
For syntax specifics see the docstrings in the [accessor directory](../../data/accessors/).

The I/O interaction is made simply of the standard Python CSV reader/writer objects.
The initialization functions for these provide the reader/writer, file, [and header].
For syntax specifics see the docstring in the [I/O file](../data_management/io.py).
