# Directory Structure
/saved_models_single_output/: Directory where trained models are saved.

/FeaVec/56dim_csv/: Directory containing the training data CSV files.

/FeaVec/T123_56dim/: Directory containing the test data CSV files.

# Machine Learning Model Evaluation and Explanation

This repository contains a Python script for evaluating and explaining machine learning models. The script performs leave-one-out cross-validation on a set of data files, evaluates the models using various metrics, and uses SHAP to provide explanations for the model's predictions.

## Requirements

Make sure you have the necessary dependencies installed. You can install them using the `requirements.txt` file:

```sh
pip install -r requirements.txt

