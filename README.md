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
* Install SpaCy model with `python -m spacy download en_core_web_sm`
* Write accessor for any additional datasets (see [`accessors/`](data/accessors/) for info)


## Usage

The usage of this work can be broken into several stages: data preparation, initial label generation, model training, 
model evaluation, and analysis.


### Data preparation

To prepare for training and evaluation several things have to be pre-computed and configured.

* Download and extract the wikipedia data with the [script](data/scripts/extract_wikipedia_corpus.py)
* Train a fastText model on a local dataset (see [GitHub for info](https://github.com/facebookresearch/fastText/)) [optional]
  * Place the trained model into `data/lexicons/fast_text/`
* Prepare the datasets for pre-processing by running their [individual scripts](data/preparation/) or all at once using 
* Execute the pre-processing [scipt](execution/pre_processing/pre_process.py)
* If you are planning on training the intent model
    * Pull subset of wikipedia document with [script](execution/pre_train/wikipedia_subset.py)
    * Generate subset of storm-front data with the [script](execution/pre_train/storm_front_subset.py)
* Specify your working dataset and fastText model in [`config.py`](config.py)
  * If you do not have a `config.py` file already then start one using a copy of [`config_template.py`](config_template.py).


### Train abuse model

* Ensure the hate speech dataset, kaggle, and insults are pre-processed
* Run the [bash script](data/scripts/generate_abuse.sh) to combine them
* Run the abuse training [script](execution/training/abuse.py)


### Train intent model

* Add the source dataset to [`config.py`](config.py)
* Run the [rough label generation script](execution/intent/compute_rough_labels.py)
* Extract the verbs from the intent frames and compute their embeddings with [collection script](execution/analysis/embeddings/collect_intent_verbs.py)
* Refine rough labels with [script](execution/intent/refine_rough_labels.py)
* Compute sequence-context matrix with [script](execution/pre_train/context_sequence_matrix.py)
* Train the model with [training script](execution/training/intent.py)


### Make predictions

Now that you have a trained abuse and intent model predictions can be made for any target dataset of interest.
This can be done by simply specifying the name of the target dataset in the [config file](config.py) and executing [the prediction script](execution/prediction/abusive_intent.py).
This will make and save a prediction for each document in the targeted corpus to `data/processed_data/[dataset_name]/analysis/intent_abuse/`

### Analysis

The analysis scripts are under [execution](execution/analysis/) and should be named/placed intuitively corresponding to how they're referred to in the thesis.


## Note on cleanliness

Most of the outdated files should have been removed by im sure unused functions and files are still here and there, ignore them.
