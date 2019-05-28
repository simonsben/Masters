# Environment

Overview of the environment and installation steps taken.

**NOTE:** the system used was running Windows 10 with Desbian WSL.

## Execution environment setup

The majority of the dependancies can be installed by simply installing `Python 3.7`, 
initializing a viritual environment, then running `pip install -r requirements.txt`.

### FastText

`FastText` was one of the packages that requires the most steps to install.
The following steps were taken to install it:

1. Clone the [repo](https://github.com/facebookresearch/fastText)
2. Build and install the project (as specified [here](https://github.com/facebookresearch/fastText/tree/master/python))
    1. **NOTE:** This required the installation of [VS C++ build tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017)
    2. When building ensure you are in your projects virtual environment, if you're using PyCharm you can simply use the 
    *Terminal* tab at the bottom to navigate into the fastText directory in order to build it.
3. Download the English `.bin` pre-trained [model](https://fasttext.cc/docs/en/crawl-vectors.html)
    1. **NOTE:** The simple `.txt` model can be used if you don't have any out-of-vocabulary (OOV) words

### SpaCy

SpaCy will be installed by executing the `pip install` listed above, however the chosen model will still have to be downloaded.
By default the one used in the code is the `en_core_web_sm` model (small english model).
It can be installed by opening the shell/command prompt, entering the virtual environment, then running `python -m spacy download en_core_web_sm`.

## Data

### Preparing and pre-processing

In this repo the document pre-processing is broken into two stages, preparing and pre-processing.
Pre-processing applies the expected processing (ex. removes symbols, capitals, etc.) using a standardized set of functions.
However, in order to do this, most of the dataset/lexicons require some degree of modification before this can be applied.

To **prepare** the data, simply execute the [`prepare.py`](../execution/pre_processing/prepare.py) script.
To **pre-process** the data, again, just execute the [`pre-process`](../execution/pre_processing/pre_process.py) script.

### Before executing

Before executing a new dataset make sure that the dataset is downloaded and the destination directories are ready.
To do this, you can simply go to the [scripts directory](../utilities/scripts), then run the 
[`prep_dataset_run.sh`](../utilities/scripts/prep_dataset_run.sh) bash script.

**NOTE:** This requires the **exact** dataset name as its only argument
