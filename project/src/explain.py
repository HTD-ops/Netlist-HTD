# src/explain.py

import shap
from .config import SELECTED_FEATURES
import matplotlib.pyplot as plt

def explain_model(model, X_train, example, selected_features, gate_name):
    """
    Explain the model prediction for a specific example using SHAP.
    
    Args:
        model (Sequential): Trained Keras model.
        X_train (ndarray): Training features used for SHAP background.
        example (ndarray): Single example to explain.
        selected_features (list): List of feature names.
        gate_name (str): Name of the gate being explained.
    """
    # Initialize SHAP DeepExplainer
    explainer = shap.DeepExplainer(model, X_train)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(example)
    
    # Print SHAP values and expected values
    print("SHAP Values:", shap_values)
    print("Expected Value:", explainer.expected_value)
    print("Selected Features:", selected_features)
    print("SHAP Values Shape:", shap_values[0].shape)
    print("Expected Value Shape:", explainer.expected_value.shape)
    
    # Visualize SHAP values using waterfall plot
    shap.initjs()
    shap.plots.waterfall(shap.Explanation(
        values=shap_values[0][:, 0],  # SHAP values for class 0
        base_values=explainer.expected_value[0].numpy(),
        data=example[0],
        feature_names=selected_features
    ))
    plt.show()
