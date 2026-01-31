import joblib
import pandas as pd
import numpy as np

# 1. Define the filename for the saved model
model_filename = 'xgboost_model.joblib'

# 2. Load the trained XGBoost model
try:
    model_xgb = joblib.load(model_filename)
    print(f"XGBoost model loaded successfully from {model_filename}")
except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found. Make sure the model is saved in the current directory.")
    exit()

# 3. Define the prediction function
def predict_fraud(data_point: pd.DataFrame) -> np.ndarray:
    """
    Predicts whether a given data point represents a fraudulent transaction.

    Args:
        data_point (pd.DataFrame): A DataFrame containing one row of features
                                   matching the training data structure (29 features, excluding 'Time' and 'Class').

    Returns:
        np.ndarray: An array containing the prediction (0 for non-fraud, 1 for fraud).
    """
    if data_point.shape[1] != model_xgb.n_features_in_:
        raise ValueError(f"Input data point must have {model_xgb.n_features_in_} features, but got {data_point.shape[1]}")
    prediction = model_xgb.predict(data_point)
    return prediction

# 4. Example Usage:
if __name__ == "__main__":
    print("\n--- Example Prediction ---")
    # Create a sample data point for prediction.
    # This should mimic the structure of X_train (29 features).
    # For demonstration, we'll use an example inspired by X_train's first row
    # or create a synthetic one. Make sure it has the same columns as X_train.
    
    # IMPORTANT: Replace this with actual new data you want to predict on.
    # Using X_train.iloc[0:1] as an example template for column names and order.
    # In a real application, you would receive new data, not from the training set.
    
    # Assuming X_train was available during model training, we recreate its column names.
    # In a deployed app, you'd define these expected columns explicitly.
    # For simplicity, we'll manually define a sample data point with 29 features.
    
    # This array represents a single new transaction's features.
    # The values below are synthetic or taken from X_test/X_train for demonstration.
    # In a real scenario, this would come from a live data source.
    sample_data_values = np.array([ 
        -1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 0.462388, 0.239599, 
        0.098698, 0.363787, 0.090794, -0.551600, -0.617801, -0.991390, -0.311169,
        1.468177, -0.470401, 0.207971, 0.025791, 0.403993, 0.251412, -0.018307, 
        0.277838, -0.110474, 0.066928, 0.128539, -0.189115, 0.133558, -0.021053, 
        149.62
    ]).reshape(1, -1)
    
    # Feature names (must match the order and names used during training)
    # Assuming `feature_names` variable was created in previous steps and contains the 29 column names.
    # If `feature_names` is not available, you would need to manually list them or ensure the input data is ordered correctly.
    # For this example, let's explicitly list the expected feature names.
    expected_feature_names = [
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 
        'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 
        'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
    ]
    
    new_data = pd.DataFrame(sample_data_values, columns=expected_feature_names)
    
    print("New data point to predict:")
    print(new_data)

    prediction = predict_fraud(new_data)

    if prediction[0] == 0:
        print("\nPrediction: This transaction is likely NON-FRAUDULENT.")
    else:
        print("\nPrediction: This transaction is likely FRAUDULENT.")
