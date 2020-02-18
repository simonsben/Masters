# Usage


## Initial setup

* [Optional] Make virtual environment for project
* Install libraries with `pip install -r requirements.txt` (when inside the virtual environment)
    * NOTE: *enter* the environment with the activate script in the venv directory
* Get datasets that you'd like to run into the `data/datasets` directory
* Run the preparation script for the dataset, if there isn't one, write it
* If there isn't an accessor written for the dataset, write one
* Run the pre-process script (`execution/pre-processing/pre-process`), ensuring that the desired datasets are in the list

