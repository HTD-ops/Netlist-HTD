# src/model_utils.py

from keras.models import Sequential, load_model
from keras.layers import Dense
from .config import SELECTED_FEATURES
import os

def build_model(input_dim):
    """
    Build a Sequential Keras model.
    
    Args:
        input_dim (int): Number of input features.
    
    Returns:
        model (Sequential): Compiled Keras model.
    """
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    """
    Train the Keras model.
    
    Args:
        model (Sequential): Keras model to train.
        X_train (ndarray): Training features.
        y_train (ndarray): Training labels.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
    
    Returns:
        history: Training history.
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return history

def save_model(model, models_dir, test_file):
    """
    Save the trained model to a file.
    
    Args:
        model (Sequential): Trained Keras model.
        models_dir (str): Directory to save the model.
        test_file (str): Test file name to associate with the model.
    """
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_filename = f'model_{test_file}.h5'
    model_path = os.path.join(models_dir, model_filename)
    model.save(model_path)
    print(f'Model saved for {test_file} at {model_path}')

def load_trained_model(models_dir, test_file):
    """
    Load a trained Keras model from file.
    
    Args:
        models_dir (str): Directory where models are saved.
        test_file (str): Test file name to associate with the model.
    
    Returns:
        model (Sequential): Loaded Keras model.
    """
    model_filename = f'model_{test_file}.h5'
    model_path = os.path.join(models_dir, model_filename)
    if not os.path.exists(model_path):
        print(f'Model file {model_path} does not exist.')
        return None
    model = load_model(model_path)
    return model
