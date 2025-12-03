import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.metrics import mean_squared_error

def load_data():
    """Loads California Housing data and creates the DataFrame."""
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    return X, y

def preprocess_data(X, y):
    """Splits data and trains/applies the StandardScaler."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def find_best_model(X_train, X_test, y_train, y_test):
    """
    Tests Linear, Ridge, and Lasso Regression models,
    finds the best one based on RMSE, and returns it.
    """

    models = {
        'LinearRegression': {
            'model': LinearRegression(),
            'params': {} 
        },
        'Ridge': {
            'model': Ridge(),
            'params': {'alpha': [0.1, 1.0, 10.0, 100.0]}
        },
        'Lasso': {
            'model': Lasso(max_iter=10000), 
            'params': {'alpha': [0.001, 0.01, 0.1, 1.0]}
        }
    }
    
    best_rmse = float('inf')
    best_model_name = ""
    best_model = None
    
    print("\n--- Starting Model Selection Process ---")
    for name, config in models.items():
        model = config['model']
        params = config['params']
        
        if params:
            grid_search = GridSearchCV(
                model, 
                params, 
                scoring='neg_mean_squared_error', 
                cv=5,
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            final_model = grid_search.best_estimator_
            print(f"Tuned {name}: Best Alpha = {grid_search.best_params_.get('alpha', 'N/A')}")
        else:
            # Train model without tuning (Linear Regression)
            model.fit(X_train, y_train)
            final_model = model
            print(f"Trained {name}.")

        # Evaluate the final model on the test set
        y_pred = final_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"{name} RMSE: {rmse:.4f}")
        
        # Check if this model is the new best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = final_model

    print(f"\n Best Model Overall: {best_model_name} with RMSE: {best_rmse:.4f}")
    return best_model, best_model_name

def save_artifacts(model, scaler, model_filename='best_house_model.joblib', scaler_filename='scaler.joblib'):
    """Saves the best trained model and scaler object."""
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"\nArtifacts Saved:")
    print(f"Model saved as '{model_filename}'")
    print(f"Scaler saved as '{scaler_filename}'")


if __name__ == '__main__':
    X, y = load_data()
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Train and select the best model
    best_model, best_model_name = find_best_model(X_train_scaled, X_test_scaled, y_train, y_test)

    # Save the trained model and scaler for the Streamlit app
    save_artifacts(best_model, scaler, model_filename='best_house_model.joblib')