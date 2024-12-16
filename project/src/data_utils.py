# src/data_utils.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
import os
from .config import SELECTED_FEATURES, SELECTED_FEATURES_FOR_TRAIN, SELECTED_FEATURES_FOR_TEST, LABEL_COLUMN

def load_data(csv_file, feature_columns, label_column=LABEL_COLUMN):
    """
    Load data from a CSV file.
    
    Args:
        csv_file (str): Path to the CSV file.
        feature_columns (list): List of feature column names.
        label_column (str): Name of the label column.
    
    Returns:
        X (DataFrame): Feature matrix.
        y (Series): Labels.
        names (ndarray): Gate names.
    """
    df = pd.read_csv(csv_file)
    X = df[feature_columns]
    y = df[label_column]
    names = df['gatename'].values
    return X, y, names

def get_positive_sample_names(y, sample_names_):
    """
    Retrieve names of positive samples.
    
    Args:
        y (Series or ndarray): Labels.
        sample_names_ (ndarray): Sample names.
    
    Returns:
        ndarray: Names of positive samples.
    """
    positive_sample_names_ = sample_names_[y == 1]
    return positive_sample_names_

def get_confusion_matrix_sample_names(y_true, y_pred_, sample_names_):
    """
    Get sample names for true positives, false negatives, and false positives.
    
    Args:
        y_true (ndarray): True labels.
        y_pred_ (ndarray): Predicted labels.
        sample_names_ (ndarray): Sample names.
    
    Returns:
        tuple: (true_positive_names, false_negative_names, false_positive_names)
    """
    if y_pred_.ndim == 2:
        y_pred_ = y_pred_.ravel()
    true_positive_indices = np.where((y_true == 1) & (y_pred_ == 1))[0]
    false_negative_indices = np.where((y_true == 1) & (y_pred_ == 0))[0]
    false_positive_indices = np.where((y_true == 0) & (y_pred_ == 1))[0]

    true_positive_sample_names = sample_names_[true_positive_indices]
    false_negative_sample_names = sample_names_[false_negative_indices]
    false_positive_sample_names = sample_names_[false_positive_indices]

    return true_positive_sample_names, false_negative_sample_names, false_positive_sample_names

def preprocess_data(train_files, test_file, train_path, selected_features):
    """
    Preprocess training and testing data.
    
    Args:
        train_files (list): List of training file names.
        test_file (str): Current test file name.
        train_path (str): Path to training data.
        selected_features (list): List of selected features.
    
    Returns:
        X_train (ndarray): Preprocessed training features.
        y_train (ndarray): Training labels.
        X_test (DataFrame): Preprocessed test features.
        y_test (ndarray): Test labels.
        sample_names (ndarray): Test sample names.
    """
    from collections import defaultdict
    import pandas as pd

    x_train_list = []
    y_train_list = []

    for train_file in train_files:
        if train_file == test_file:
            continue

        X, y, _ = load_data(os.path.join(train_path, train_file), selected_features)
        data = pd.concat([X, y], axis=1).drop_duplicates(keep='first')

        # Separate Trojan and Normal nodes
        Trojan = data[data[LABEL_COLUMN] == 1]
        Normal = data[data[LABEL_COLUMN] == 0]

        # Balance the data by replicating Trojan samples
        if len(Trojan) > 0:
            Trojan = pd.concat([Trojan] * 3, ignore_index=True)

        # Combine and apply SMOTEENN
        data = pd.concat([Trojan, Normal], ignore_index=True)
        smenn = SMOTEENN()
        X_res, y_res = smenn.fit_resample(data[selected_features], data[LABEL_COLUMN])

        x_train_list.append(X_res)
        y_train_list.append(y_res)

    # Combine all training data
    X_train = pd.concat(x_train_list, ignore_index=True).values
    y_train = np.concatenate(y_train_list)

    return X_train, y_train
