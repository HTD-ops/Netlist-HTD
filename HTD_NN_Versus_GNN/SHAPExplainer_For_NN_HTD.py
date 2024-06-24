import os
import pandas as pd
import numpy as np
import shap
import time
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# Model directory path
models_dir = 'saved_models_single_output'

# Training data directory path for standardization
train_relative_path = '/FeaVec/56dim_csv/'

# Test data directory path
test_relative_path = '/FeaVec/T123_56dim/'

# File names for recording the results
file_names_ = [
    'RS232-T1000.csv', 'RS232-T1100.csv', 'RS232-T1200.csv',
    'RS232-T1300.csv', 'RS232-T1400.csv', 'RS232-T1500.csv',
    'RS232-T1600.csv',
    's38417-T100.csv', 's38417-T200.csv', 's38417-T300.csv',
    's35932-T100.csv', 's35932-T200.csv', 's35932-T300.csv',
]

# Keys for result metrics
keys_ = ["TPR", "TNR", "cfm", "TP", "FN", "FP"]
results_all = {name: {key: None for key in keys_} for name in file_names_}

# Get all training file names
file_path = os.getcwd() + train_relative_path
files = [f for f in os.listdir(file_path) if f.endswith('.csv')]

# Get all test file names
to_test_path = os.getcwd() + test_relative_path
to_test_files = [f for f in os.listdir(to_test_path) if f.endswith('.csv')]

# Feature selection
files_tested = []
# feature_select_or_not = 1
selected_features = ['fan_in_1', 'fan_in_2', 'fan_in_3', 'fan_in_4', 'fan_in_5',
                     'in_flipflop_1', 'in_flipflop_4', 'in_flipflop_5',
                     'in_near_flipflop', 'out_near_flipflop', 'DPI', 'DPO',
                     'loop_3', 'gate_in_1', 'gate_in_2', 'gate_in_3', 'gate_in_5',
                     'inv_3', 'inv_4', 'inv_5']

selected_features_for_train = selected_features + ['label']
selected_features_for_test = ['gatename'] + selected_features

# Function to load data from CSV files
def load_data(csv_file, feature_columns, label_column='label'):
    df = pd.read_csv(csv_file)
    X = df[feature_columns]
    y = df[label_column]
    names = df['gatename'].values
    return X, y, names

# Function to get confusion matrix sample names
def get_confusion_matrix_sample_names(y_true, y_pred_, sample_names_):
    if y_pred_.ndim == 2:
        y_pred_ = y_pred_.ravel()
    true_positive_indices = np.where((y_true == 1) & (y_pred_ == 1))[0]
    false_negative_indices = np.where((y_true == 1) & (y_pred_ == 0))[0]
    false_positive_indices = np.where((y_true == 0) & (y_pred_ == 1))[0]

    true_positive_sample_names = sample_names_[true_positive_indices]
    false_negative_sample_names = sample_names_[false_negative_indices]
    false_positive_sample_names = sample_names_[false_positive_indices]

    return true_positive_sample_names, false_negative_sample_names, false_positive_sample_names

# Function to get positive sample names
def get_positive_sample_names(y, sample_names_):
    positive_sample_names_ = sample_names_[y == 1]
    return positive_sample_names_

label_column = 'label'

# Metrics for calculating averages
total_acc = 0
total_precision = 0
total_recall = 0
total_f1 = 0
total_TNR = 0

# Perform leave-one-out cross-validation
tt1 = time.time()
for i, test_file in enumerate(to_test_files):

    print('\n-------({}/{}) Current test set: {}-------'.format(i + 1, len(files), test_file))

    # Load test data
    x_test, y_test, sample_names = load_data(os.path.join(to_test_path, test_file), selected_features_for_test)
    x_test_raw = x_test.copy(deep=True)
    positive_sample_names = get_positive_sample_names(y_test, sample_names)
    print(f"Number of positive samples in test set: {len(positive_sample_names)}\n{positive_sample_names}")

    # Initialize containers for training data
    x_train_list = []
    y_train_list = []

    # Aggregate training data from all other files
    for j, train_file in enumerate(files):
        if i == j:  # Skip the test file
            continue

        # Load train data
        X, y, _ = load_data(os.path.join(file_path, train_file), selected_features)

        # Drop duplicates
        data = pd.concat([X, y], axis=1).drop_duplicates(keep='first')

        # Separate Trojan and Normal nodes
        Trojan = data.loc[data[data.columns[-1]] == 1]
        Normal = data.loc[data[data.columns[-1]] == 0]

        # Ensure Trojan instances are at least greater than 6 before SMOTEENN
        if len(Trojan) < 7:
            Trojan = pd.concat([Trojan] * (7 // len(Trojan) + 1), ignore_index=True)

        # Combine Trojan and Normal data
        data = pd.concat([Trojan, Normal], ignore_index=True)

        # Apply SMOTEENN
        smenn = SMOTEENN()
        X_res, y_res = smenn.fit_resample(data[selected_features], data[label_column])

        x_train_list.append(X_res)
        y_train_list.append(y_res)

    # Combine all training data
    x_train = np.vstack(x_train_list)
    y_train = np.concatenate(y_train_list)

    # Data preprocessing: Standardization
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test[selected_features] = scaler.transform(x_test[selected_features].values)

    # Get standardization parameters
    mean_values = scaler.mean_
    std_values = scaler.scale_

    # Load model and predict
    model_path = os.path.join(models_dir, f'model_{test_file}.h5')

    if not os.path.exists(model_path):
        print(f'Model file {model_path} does not exist. Skipping.')
        raise FileNotFoundError(f"No model file found at {model_path}")
        continue

    model = load_model(model_path)

    all_explain = ['U305']
    column_name = 'gatename'
    for gatename_to_explain in all_explain:
        if gatename_to_explain != 'U305':
            continue

        print("Explaining gate:", gatename_to_explain)

        if gatename_to_explain in x_test[column_name].values:
            count = (x_test[column_name] == gatename_to_explain).sum()
            if count == 1:
                example = x_test.iloc[np.where(x_test['gatename'] == gatename_to_explain)[0][0], 1:].values
                example = example.astype(np.float32).reshape(1, -1)
            else:
                raise ValueError(f"{gatename_to_explain} appears {count} times in column {column_name}, which is not unique.")
        else:
            raise ValueError(f"{gatename_to_explain} is not in column {column_name}")

        explainer = shap.DeepExplainer(model, x_train)
        shap_values = explainer.shap_values(example)

        raw_example = x_test_raw.iloc[np.where(x_test_raw['gatename'] == gatename_to_explain)[0][0], 1:].values

        shap.initjs()
        shap.plots.waterfall(shap.Explanation(values=shap_values[0].squeeze(),
                                              base_values=explainer.expected_value[0].numpy(),
                                              data=raw_example,
                                              feature_names=selected_features))

    x_gatename = x_test.iloc[:, 1:].values
    x_test = x_test.iloc[:, 1:].values

    y_pred_raw = model.predict(x_test)
    y_pred = y_pred_raw > 0.5

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)  # TPR
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    tnr = tn / (tn + fp)

    # Accumulate metrics
    total_acc += acc
    total_precision += precision
    total_recall += recall
    total_f1 += f1
    total_TNR += tnr
    files_tested.append(test_file)

    true_positive_names, false_negative_names, false_positive_names = \
        get_confusion_matrix_sample_names(y_test, y_pred, sample_names)
    print("Correctly classified positive samples:")
    print(true_positive_names)
    print("False negative samples:")
    if false_negative_names.size > 0:
        print(false_negative_names)
    else:
        print("FN=0, all positive samples are correctly classified, TPR=100%")
    print("False positive samples:")
    print(false_positive_names)

    print("Confusion Matrix:\n", cm)
    print("Accuracy: {:.2f}%".format(acc * 100))
    print("Precision: {:.2f}%".format(precision * 100))
    print("F1 Score: {:.2f}%".format(f1 * 100))
    print("Recall (TPR): {:.2f}%".format(recall * 100))
    print("TNR: {:.2f}%".format(tnr * 100))

    results_all[test_file]["TPR"] = "Recall (TPR): {:.2f}%".format(recall * 100)
    results_all[test_file]["TNR"] = "TNR: {:.2f}%".format(tnr * 100)
    results_all[test_file]["cfm"] = cm
    results_all[test_file]["TP"] = true_positive_names
    results_all[test_file]["FN"] = false_negative_names
    results_all[test_file]["FP"] = false_positive_names

# Calculate and print average metrics
n = len(files_tested)
avg_acc = total_acc / n
avg_precision = total_precision / n
avg_recall = total_recall / n  # Average TPR
avg_f1 = total_f1 / n
avg_tnr = total_TNR / n

print('\nAverage metrics over all test files:')
print(f'files_tested:{len(files_tested)}\n{files_tested}')
print("Average Accuracy: {:.2f}%".format(avg_acc * 100))
print("Average Precision: {:.2f}%".format(avg_precision * 100))
print("Average F1 Score: {:.2f}%".format(avg_f1 * 100))
print("Average Recall (TPR): {:.2f}%".format(avg_recall * 100))
print("Average TNR: {:.2f}%".format(avg_tnr * 100))
tt2 = time.time()
print('Total time: {:.2f} seconds'.format(tt2 - tt1))
