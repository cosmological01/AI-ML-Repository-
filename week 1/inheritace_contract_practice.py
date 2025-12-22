import abc

# --- 1. ABSTRACT BASE CLASS (The Blueprint/Contract) ---

class MLModel(abc.ABC):
    """
    Defines the required methods (the contract) for ANY model 
    (Linear, Neural Network, Decision Tree) in our system.
    """
    
    def __init__(self, name):
        # All models have a name
        self.model_name = name
        print(f"Model {self.model_name} initialized.")

    # A simple, shared method implemented in the Base Class
    def evaluate(self, X_test, y_test):
        """
        Public method to evaluate performance (common to all models).
        Calls the specialized predict method.
        """
        predictions = self.predict(X_test)
        # Placeholder for real evaluation metrics
        print(f"[{self.model_name}] Evaluation complete. Metrics Placeholder.")
        return predictions

    # ABSTRACT METHOD 1: Must be implemented by subclasses
    @abc.abstractmethod
    def fit(self, X_train, y_train):
        """Must define HOW the model learns from data."""
        pass

    # ABSTRACT METHOD 2: Must be implemented by subclasses
    @abc.abstractmethod
    def predict(self, X):
        """Must define HOW the model makes predictions."""
        pass


# --- 2. CONCRETE DERIVED CLASS: Decision Tree ---

class DecisionTreeModel(MLModel):
    """
    A concrete model class that MUST fulfill the contract (fit and predict).
    """
    def __init__(self):
        super().__init__("DecisionTree") # Initialize the base class
        self.trained = False

    # Implementation for the required 'fit' method
    def fit(self, X_train, y_train):
        # Placeholder: This is where the tree building algorithm would go
        print(f"[{self.model_name}] Training started: Building the tree structure...")
        self.trained = True

    # Implementation for the required 'predict' method
    def predict(self, X):
        if not self.trained:
            print(f"[{self.model_name}] Warning: Model not trained yet!")
            return None
        # Placeholder: This is where the prediction logic traverses the tree
        print(f"[{self.model_name}] Predicting outputs for {len(X)} samples.")
        return [0] * len(X) # Returns list of placeholder predictions

# --- USAGE ---

# Dummy Data (Simple list of samples)
dummy_data = [1, 2, 3, 4]

# 1. Instantiate the concrete model
tree_model = DecisionTreeModel()

# 2. Call the required fit method
tree_model.fit(dummy_data, dummy_data) 

# 3. Call the shared evaluate method
results = tree_model.evaluate(dummy_data, dummy_data) 

# Example of ABC enforcement:
# If you tried to create a class that inherited from MLModel but didn't implement 
# 'fit' and 'predict', Python would raise an error.
# m_broken = MLModel() # This would raise TypeError!