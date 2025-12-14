import abc
import numpy as np

# --- 1. ABSTRACT BASE CLASS (The Model Contract) ---

class MLModel(abc.ABC):
    """
    Abstract Base Class for all Machine Learning Models.
    It enforces that every concrete model must define its key methods.
    """
    
    def __init__(self, learning_rate=0.01):
        # Common attributes for all models
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        print(f"[{self.__class__.__name__}] Model initialized with LR: {learning_rate}")

    # The Template Method: The required execution flow for training
    def train(self, X_train, y_train, epochs=100):
        """Defines the fixed pipeline for model training."""
        print(f"[{self.__class__.__name__}] Starting training for {epochs} epochs...")
        
        self._initialize_parameters(X_train.shape[1])
        
        for epoch in range(epochs):
            loss = self._forward_pass(X_train, y_train)
            self._backward_pass(X_train, y_train)
            if (epoch + 1) % (epochs // 5) == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")
        
        print(f"[{self.__class__.__name__}] Training complete.")

    # ABSTRACT METHODS (Must be implemented by concrete models)
    @abc.abstractmethod
    def _initialize_parameters(self, n_features):
        """Initializes weights/biases specific to the model type."""
        pass
    
    @abc.abstractmethod
    def _forward_pass(self, X, y):
        """Calculates predictions and returns the loss."""
        pass

    @abc.abstractmethod
    def _backward_pass(self, X, y):
        """Calculates gradients and updates parameters."""
        pass

    @abc.abstractmethod
    def predict(self, X):
        """Generates final output predictions."""
        pass


# --- 2. CONCRETE DERIVED CLASS: LINEAR REGRESSION MODEL ---

class LinearRegressionModel(MLModel):
    """
    Concrete implementation of a simple Linear Regression model.
    It fulfills the contract defined by MLModel.
    """
    
    def _initialize_parameters(self, n_features):
        # Initialize weights randomly, bias to zero
        np.random.seed(42)
        self.weights = np.random.randn(n_features, 1) * 0.01
        self.bias = 0
        print("[LinearRegressionModel] Parameters initialized.")
    
    def _forward_pass(self, X, y):
        # Calculate y_predicted
        y_pred = np.dot(X, self.weights) + self.bias
        # Mean Squared Error (MSE) Loss
        loss = np.mean((y_pred - y) ** 2) / 2
        
        # Store prediction error for backward pass (self is the common practice)
        self._y_pred = y_pred 
        self._X = X 
        self._y = y
        return loss

    def _backward_pass(self, X, y):
        # Simple gradient calculation for MSE loss
        m = X.shape[0]
        error = self._y_pred - self._y
        
        # Calculate gradients
        dw = np.dot(self._X.T, error) / m
        db = np.sum(error) / m
        
        # Update parameters
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
        
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# --- USAGE ---

# 1. Generate dummy data (simulating a dataset)
# 100 samples, 2 features
X_dummy = np.random.rand(100, 2)
# Target variable is a linear combination of features plus noise
y_dummy = 2 * X_dummy[:, 0] + 5 * X_dummy[:, 1] + np.random.randn(100) * 0.1
y_dummy = y_dummy.reshape(-1, 1)

# 2. Instantiate and use the model
model = LinearRegressionModel(learning_rate=0.05)

# 3. Call the public 'train' template method
model.train(X_dummy, y_dummy, epochs=500)

# 4. Make a prediction
new_data = np.array([[0.5, 0.5]])
prediction = model.predict(new_data)

print(f"\nPrediction for {new_data}: {prediction[0][0]:.4f}")
# Expected output is close to 2*0.5 + 5*0.5 = 3.5