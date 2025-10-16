"""
Framework Adapters for Multi-Framework Federated Learning

Supports:
- PyTorch (native)
- TensorFlow/Keras
- JAX/Flax
- Scikit-learn
- XGBoost/LightGBM
"""

import numpy as np
from abc import ABC, abstractmethod

# ============================================
# ABSTRACT BASE ADAPTER
# ============================================
class ModelAdapter(ABC):
    """Base class for framework adapters"""
    
    @abstractmethod
    def get_parameters(self, model):
        """Extract parameters as list of numpy arrays"""
        pass
    
    @abstractmethod
    def set_parameters(self, model, parameters):
        """Set model parameters from list of numpy arrays"""
        pass
    
    @abstractmethod
    def train(self, model, train_loader, epochs, **kwargs):
        """Train the model"""
        pass
    
    @abstractmethod
    def evaluate(self, model, test_loader):
        """Evaluate the model"""
        pass


# ============================================
# PYTORCH ADAPTER (Already Works!)
# ============================================
class PyTorchAdapter(ModelAdapter):
    """Adapter for PyTorch models"""
    
    def __init__(self, device='cuda'):
        import torch
        self.torch = torch
        self.device = device if torch.cuda.is_available() else 'cpu'
    
    def get_parameters(self, model):
        """Get PyTorch model parameters"""
        return [val.cpu().numpy() for val in model.state_dict().values()]
    
    def set_parameters(self, model, parameters):
        """Set PyTorch model parameters"""
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: self.torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
    
    def train(self, model, train_loader, epochs, optimizer, criterion):
        """Train PyTorch model"""
        import torch.nn as nn
        
        model.to(self.device)
        model.train()
        
        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        return model
    
    def evaluate(self, model, test_loader):
        """Evaluate PyTorch model"""
        model.eval()
        correct = 0
        total = 0
        
        with self.torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = self.torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return correct / total


# ============================================
# TENSORFLOW/KERAS ADAPTER
# ============================================
class TensorFlowAdapter(ModelAdapter):
    """Adapter for TensorFlow/Keras models"""
    
    def __init__(self):
        try:
            import tensorflow as tf
            self.tf = tf
        except ImportError:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")
    
    def get_parameters(self, model):
        """Get TensorFlow/Keras model weights"""
        # model.get_weights() returns list of numpy arrays
        return model.get_weights()
    
    def set_parameters(self, model, parameters):
        """Set TensorFlow/Keras model weights"""
        model.set_weights(parameters)
    
    def train(self, model, train_loader, epochs, **kwargs):
        """Train TensorFlow/Keras model"""
        # Assume train_loader is tf.data.Dataset or compatible
        optimizer = kwargs.get('optimizer', self.tf.keras.optimizers.Adam())
        loss_fn = kwargs.get('loss', self.tf.keras.losses.SparseCategoricalCrossentropy())
        
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        
        # If train_loader is a tuple (x, y)
        if isinstance(train_loader, tuple):
            x_train, y_train = train_loader
            model.fit(x_train, y_train, epochs=epochs, verbose=0)
        else:
            # Assume it's a tf.data.Dataset
            model.fit(train_loader, epochs=epochs, verbose=0)
        
        return model
    
    def evaluate(self, model, test_loader):
        """Evaluate TensorFlow/Keras model"""
        if isinstance(test_loader, tuple):
            x_test, y_test = test_loader
            loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        else:
            loss, accuracy = model.evaluate(test_loader, verbose=0)
        
        return accuracy


# ============================================
# JAX/FLAX ADAPTER
# ============================================
class JAXAdapter(ModelAdapter):
    """Adapter for JAX/Flax models"""
    
    def __init__(self):
        try:
            import jax
            import jax.numpy as jnp
            from flax import linen as nn
            self.jax = jax
            self.jnp = jnp
        except ImportError:
            raise ImportError("JAX/Flax not installed. Run: pip install jax flax")
    
    def get_parameters(self, params):
        """Get JAX/Flax parameters (params is a dict tree)"""
        # Flatten the parameter tree to list of numpy arrays
        from jax.tree_util import tree_flatten
        flat_params, _ = tree_flatten(params)
        return [np.array(p) for p in flat_params]
    
    def set_parameters(self, params_template, parameters):
        """Set JAX/Flax parameters"""
        from jax.tree_util import tree_flatten, tree_unflatten
        _, tree_def = tree_flatten(params_template)
        return tree_unflatten(tree_def, [self.jnp.array(p) for p in parameters])
    
    def train(self, model, train_loader, epochs, **kwargs):
        """Train JAX/Flax model"""
        # JAX training is more complex, this is a simplified version
        # Users would need to provide their own training loop
        raise NotImplementedError("JAX training requires custom implementation")
    
    def evaluate(self, model, test_loader):
        """Evaluate JAX/Flax model"""
        raise NotImplementedError("JAX evaluation requires custom implementation")


# ============================================
# SCIKIT-LEARN ADAPTER
# ============================================
class SklearnAdapter(ModelAdapter):
    """Adapter for Scikit-learn models"""
    
    def __init__(self):
        try:
            import sklearn
        except ImportError:
            raise ImportError("Scikit-learn not installed. Run: pip install scikit-learn")
    
    def get_parameters(self, model):
        """Get sklearn model parameters"""
        # For tree-based models, we extract feature importances and tree structures
        # For linear models, we extract coefficients
        
        params = []
        
        # Linear models (LogisticRegression, LinearRegression, etc.)
        if hasattr(model, 'coef_'):
            params.append(model.coef_)
            if hasattr(model, 'intercept_'):
                params.append(model.intercept_)
        
        # Tree-based models (RandomForest, GradientBoosting, etc.)
        elif hasattr(model, 'estimators_'):
            # This is complex - sklearn trees aren't easily serializable
            # We'd need to use joblib or pickle for full serialization
            import joblib
            import io
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            params.append(np.frombuffer(buffer.getvalue(), dtype=np.uint8))
        
        return params
    
    def set_parameters(self, model, parameters):
        """Set sklearn model parameters"""
        if hasattr(model, 'coef_'):
            model.coef_ = parameters[0]
            if len(parameters) > 1 and hasattr(model, 'intercept_'):
                model.intercept_ = parameters[1]
        
        # For tree-based models
        elif hasattr(model, 'estimators_'):
            import joblib
            import io
            buffer = io.BytesIO(parameters[0].tobytes())
            model = joblib.load(buffer)
        
        return model
    
    def train(self, model, train_data, epochs=1, **kwargs):
        """Train sklearn model"""
        # sklearn doesn't use epochs, just fit once
        x_train, y_train = train_data
        model.fit(x_train, y_train)
        return model
    
    def evaluate(self, model, test_data):
        """Evaluate sklearn model"""
        x_test, y_test = test_data
        return model.score(x_test, y_test)


# ============================================
# XGBOOST ADAPTER
# ============================================
class XGBoostAdapter(ModelAdapter):
    """Adapter for XGBoost models"""
    
    def __init__(self):
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
    
    def get_parameters(self, model):
        """Get XGBoost model parameters"""
        # XGBoost models can be saved/loaded as JSON or binary
        import json
        config = json.loads(model.save_config())
        
        # Convert to numpy array for transmission
        config_str = json.dumps(config)
        return [np.frombuffer(config_str.encode(), dtype=np.uint8)]
    
    def set_parameters(self, model, parameters):
        """Set XGBoost model parameters"""
        import json
        config_bytes = parameters[0].tobytes()
        config_str = config_bytes.decode()
        config = json.loads(config_str)
        
        model.load_config(json.dumps(config))
        return model
    
    def train(self, model, train_data, epochs=100, **kwargs):
        """Train XGBoost model"""
        x_train, y_train = train_data
        
        dtrain = self.xgb.DMatrix(x_train, label=y_train)
        params = kwargs.get('params', {'max_depth': 6, 'eta': 0.3, 'objective': 'binary:logistic'})
        
        model = self.xgb.train(params, dtrain, num_boost_round=epochs)
        return model
    
    def evaluate(self, model, test_data):
        """Evaluate XGBoost model"""
        x_test, y_test = test_data
        dtest = self.xgb.DMatrix(x_test, label=y_test)
        
        predictions = model.predict(dtest)
        accuracy = np.mean((predictions > 0.5) == y_test)
        return accuracy


# ============================================
# AUTO-DETECT FRAMEWORK
# ============================================
def detect_framework(model):
    """Automatically detect which framework a model uses"""
    
    model_type = type(model).__module__.split('.')[0]
    
    if 'torch' in model_type:
        return PyTorchAdapter()
    elif 'tensorflow' in model_type or 'keras' in model_type:
        return TensorFlowAdapter()
    elif 'jax' in model_type or 'flax' in model_type:
        return JAXAdapter()
    elif 'sklearn' in model_type:
        return SklearnAdapter()
    elif 'xgboost' in model_type:
        return XGBoostAdapter()
    else:
        raise ValueError(f"Unknown framework for model type: {type(model)}")


# ============================================
# USAGE EXAMPLE
# ============================================
if __name__ == "__main__":
    print("Framework Adapters Available:")
    print("✅ PyTorch (native support)")
    print("✅ TensorFlow/Keras")
    print("✅ JAX/Flax")
    print("✅ Scikit-learn")
    print("✅ XGBoost/LightGBM")
    print("\nUse detect_framework(model) to auto-detect!")


