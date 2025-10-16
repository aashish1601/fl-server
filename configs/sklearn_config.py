"""
Configuration for Scikit-learn Models
Example: MNIST with Logistic Regression
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
import numpy as np

# ============================================
# MODEL DEFINITION (Scikit-learn)
# ============================================

# Option 1: Logistic Regression (works well with FL)
MODEL_CLASS = LogisticRegression
MODEL_KWARGS = {
    'max_iter': 1000,
    'solver': 'lbfgs',
    'multi_class': 'multinomial',
    'random_state': 42
}

# Option 2: Random Forest (uncomment to use)
# MODEL_CLASS = RandomForestClassifier
# MODEL_KWARGS = {
#     'n_estimators': 100,
#     'max_depth': 10,
#     'random_state': 42
# }

# Specify framework
FRAMEWORK = 'sklearn'

# ============================================
# TRAINING CONFIGURATION
# ============================================
# sklearn doesn't use epochs in the same way
EPOCHS_PER_ROUND = 1  # sklearn.fit() is called once
BATCH_SIZE = None  # Not applicable for sklearn

# ============================================
# DATA LOADING
# ============================================
def get_data_loaders(client_id=0):
    """
    Load MNIST data for sklearn
    Returns: ((x_train, y_train), (x_test, y_test))
    """
    # Load MNIST from sklearn
    print("Loading MNIST from sklearn...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    
    X, y = mnist['data'].to_numpy(), mnist['target'].to_numpy().astype(int)
    
    # Normalize
    X = X / 255.0
    
    # Split train/test (first 60k train, last 10k test)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    # Split between clients
    if client_id == 0:
        train_start, train_end = 0, 30000
        test_start, test_end = 0, 5000
    else:
        train_start, train_end = 30000, 60000
        test_start, test_end = 5000, 10000
    
    X_train_client = X_train[train_start:train_end]
    y_train_client = y_train[train_start:train_end]
    X_test_client = X_test[test_start:test_end]
    y_test_client = y_test[test_start:test_end]
    
    train_data = (X_train_client, y_train_client)
    test_data = (X_test_client, y_test_client)
    
    print(f"Client {client_id}: {len(X_train_client)} train, {len(X_test_client)} test samples")
    
    return train_data, test_data


# ============================================
# NOTE: Sklearn in Federated Learning
# ============================================
"""
Sklearn models work differently than deep learning models:

1. Most sklearn models don't support incremental learning
2. Parameters are extracted differently (coefficients, not gradients)
3. Averaging works well for linear models (LogisticRegression, SVM)
4. Tree-based models are harder to federate (requires special techniques)

Best sklearn models for FL:
✅ LogisticRegression - works great
✅ SGDClassifier - supports partial_fit()
✅ Linear models - easy to average
⚠️ Random Forest - complex, needs special handling
⚠️ GradientBoosting - difficult to federate

Usage:
    python server_generic.py --config configs/sklearn_config.py --num-rounds 5
    python client_generic.py --config configs/sklearn_config.py
"""


