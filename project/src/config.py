# src/config.py

import os

# Directory paths
BASE_DIR = os.getcwd()
TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'data', 'train')  # e.g., './data/train/'
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data', 'test')    # e.g., './data/test/'
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')        # Ensure this directory exists

# Feature selection
SELECTED_FEATURES = [
    'fan_in_1', 'fan_in_2', 'fan_in_3', 'fan_in_4', 'fan_in_5',
    'in_flipflop_1', 'in_flipflop_4', 'in_flipflop_5',
    'in_near_flipflop', 'out_near_flipflop', 'DPI', 'DPO',
    'loop_3', 'gate_in_1', 'gate_in_2', 'gate_in_3', 'gate_in_5',
    'inv_3', 'inv_4', 'inv_5'
]

SELECTED_FEATURES_FOR_TRAIN = SELECTED_FEATURES + ['label']  # Include label for training
SELECTED_FEATURES_FOR_TEST = ['gatename'] + SELECTED_FEATURES  # Include gate name for testing

LABEL_COLUMN = 'label'
