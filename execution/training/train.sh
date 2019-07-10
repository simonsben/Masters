#!/usr/bin/env bash

echo "Activating virtual environment"
source ../../venv/Scripts/activate

echo "Checking python version"
python --version

#echo "Training derived"
#python derived_models.py
#
#echo "Training emotion"
#python emotion_models.py
#
#echo "Training lexicon models"
#python lexicon_models.py
#
#echo "Running XGBoost predictions"
#python ../prediction/xgboost.py
#
#echo "Training stacked predictor"
#python stacked.py
#
#echo "Running stacked predictions"
#python ../prediction/stacked.py
#
#echo "Training complete."
