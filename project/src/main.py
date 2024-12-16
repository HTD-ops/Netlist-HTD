# src/main.py

import os
import numpy as np
import pandas as pd
import time

from config import TRAIN_DATA_PATH, TEST_DATA_PATH, MODELS_DIR, SELECTED_FEATURES_FOR_TRAIN, SELECTED_FEATURES_FOR_TEST, LABEL_COLUMN
from data_utils import load_data, get_positive_sample_names, get_confusion_matrix_sample_names, preprocess_data
from model_utils import build_model, train_model, save_model, load_trained_model
from evaluation import evaluate_model, print_evaluation
from explain import explain_model

def main():
    # Retrieve all CSV files in the training directory
    train_files = [f for f in os.listdir(TRAIN_DATA_PATH) if f.endswith('.csv')]
    n = len(train_files)  # Number of files

    # Initialize evaluation metrics
    total_acc = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    # Start Leave-One-Out Cross-Validation
    tt1 = time.time()
    for i, test_file in enumerate(train_files):
        # Uncomment the following lines to perform complete leave-one-out cross-validation
        # and comment out the condition below.
        # Currently, it processes only 'test_file.csv' for demonstration.

        # if test_file != 'test_file.csv':
        #     continue

        print('\n-------({}/{}) Current test set: {} -------'.format(i + 1, n, test_file))

        # Load test data and retrieve sample names
        X_test, y_test, sample_names = load_data(os.path.join(TRAIN_DATA_PATH, test_file), SELECTED_FEATURES_FOR_TEST)
        X_test_raw = X_test.copy()
        positive_sample_names = get_positive_sample_names(y_test, sample_names)
        print(type(positive_sample_names), positive_sample_names.shape, positive_sample_names.size)
        print(f"Number of positive gate node names in test set: {len(positive_sample_names)}\n{positive_sample_names}")

        # Preprocess training data
        X_train, y_train = preprocess_data(train_files, test_file, TRAIN_DATA_PATH, SELECTED_FEATURES)

        # Data preprocessing: Standardization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test[selected_features] = scaler.transform(X_test[selected_features].values)

        print(f'\n--- Loading Model and Predicting for {test_file} ---')

        # Load or train the model
        model = load_trained_model(MODELS_DIR, test_file)
        if model is None:
            # If model does not exist, build and train a new model
            model = build_model(input_dim=X_train.shape[1])
            train_model(model, X_train, y_train, epochs=10, batch_size=32)
            save_model(model, MODELS_DIR, test_file)

        # Example gate nodes to explain
        for gatename_to_explain in ['U3']:  # Modify this list as needed
            if gatename_to_explain not in sample_names:
                print(f"Gate name {gatename_to_explain} not found in test set.")
                continue

            print("Explaining gate:", gatename_to_explain)
            example_idx = np.where(sample_names == gatename_to_explain)[0]
            if len(example_idx) == 0:
                print(f"No samples found for gate {gatename_to_explain}.")
                continue

            example = X_test.iloc[example_idx[0], 1:].values
            example = example.astype(np.float32).reshape(1, -1)

            prediction = model.predict(example)
            print(gatename_to_explain, "Prediction:", prediction)
            print("Sample features:\n", example, example.shape)

            # Explain the model prediction
            explain_model(model, X_train, example, SELECTED_FEATURES, gatename_to_explain)

        # Convert test data for evaluation
        X_test_values = X_test.iloc[:, 1:].values

        # Model evaluation
        y_pred_prob = model.predict(X_test_values).ravel()
        y_pred_binary = (y_pred_prob >= 0.5).astype(int)  # Convert probabilities to binary labels using threshold 0.5
        y_test_binary = y_test.values

        # Calculate evaluation metrics
        metrics = evaluate_model(y_test_binary, y_pred_binary)

        # Accumulate metrics
        total_acc += metrics['accuracy']
        total_precision += metrics['precision']
        total_recall += metrics['recall']
        total_f1 += metrics['f1']

        # Retrieve confusion matrix sample names
        true_positive_names, false_negative_names, false_positive_names = \
            get_confusion_matrix_sample_names(y_test_binary, y_pred_binary, sample_names)
        print("Correctly classified positive sample names:")
        print(true_positive_names)
        print("False negative sample names:")
        if false_negative_names.size > 0:
            print(false_negative_names)
        else:
            print("FN=0, all positive samples correctly classified, TPR=100%")
        print("False positive sample names:")
        print(false_positive_names)

        # Print confusion matrix and evaluation metrics
        print_evaluation(metrics)

    # Calculate and print average metrics
    avg_acc = total_acc / n
    avg_precision = total_precision / n
    avg_recall = total_recall / n  # Average TPR
    avg_f1 = total_f1 / n

    print('\nAverage metrics over all test files:')
    print("Average Accuracy: {:.2f}%".format(avg_acc * 100))
    print("Average Precision: {:.2f}%".format(avg_precision * 100))
    print("Average Recall (TPR): {:.2f}%".format(avg_recall * 100))
    print("Average F1 Score: {:.2f}%".format(avg_f1 * 100))

    tt2 = time.time()
    print('Total time: {:.2f} seconds'.format(tt2 - tt1))

if __name__ == "__main__":
    main()
