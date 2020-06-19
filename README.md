![](banner.png)

This is a set of work on intent detection, specifically focused on abusive intent.
The work was done as part of my Master's at Queen's University under Professor Skillicorn.
The abusive language detection is the continuation of work done by Hannah LeBlanc.

**NOTE**: Almost all of the data sets were made by someone else, their sources should all be tagged.

## Setup

To setup the repo 

* Install Python 3.X (at this point <= 3.7 due to TensorFlow support)
* Generate a virtual environment for the project [optional]
* Install python dependencies with `pip install -r requirements.txt`
* Install SpaCy model with `python -m spacy download en_core_web_md`
* Write accessor for selected dataset (see [`accessors/`](data/accessors/) for info)


## Usage

**TODO:** Write usage things.... and make clean enough to be usable..

### Prepare data

### Train intent model

* Add the source dataset to [`config.py`](config.py)
* Run the [rough label generation script](execution/intent/compute.py)
* Compute english mask over the contexts with [script](execution/pre_train/english_mask.py)
* Extract the verbs from the intent frames and compute their embeddings with [collection script](execution/analysis/embeddings/collect_intent_verbs.py)
* Refine rough labels with [script](execution/intent/refine_initial_mask.py)
* Train the model with [training script](execution/training/intent.py)
