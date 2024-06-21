import pandas as pd
import seaborn as sns
import numpy as np
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt



def preprocess_data(train_df):
    # drop the saleprice to seperate the target variable SalePrice
    X = train_df.drop(['SalePrice'], axis=1)
    y = train_df['SalePrice']

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # Change NA numerical values to the mean value of the feautre
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Change NA categorical values to the most_frequent value of the feautre 
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine changes togther after NA's value are changed
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return X, y, preprocessor

def split_data(X, y):
    # split data into training and validation sets 
    num = random.randint(1, 100)
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=num)
    return X_train, X_valid, y_train, y_valid, num

def train_model(X_train, y_train, preprocessor, num):
    # Train rf and gb models using sklearn Pipeline function
    rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', RandomForestRegressor(n_estimators=100, random_state=num))])
    gb_model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', GradientBoostingRegressor(n_estimators=100, random_state=num))])
    
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    
    return rf_model, gb_model

def evaluate_model(model, X_valid, y_valid):
    preds = model.predict(X_valid)
    # find root mean squared error (RMSE), setting squared to false allows that
    # FutureWarning: 'squared' is deprecated in version 1.4 and will be removed in 1.6. To calculate the root mean squared error, use the function'root_mean_squared_error'.
    rmse = mean_squared_error(y_valid, preds, squared=False)
    return rmse


# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
def hyperparameter_tuning(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_estimator_



def save_submission(test_df, predictions, filename):
    submission = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': predictions})
    submission.to_csv(filename, index=False)


def plot_results(y_valid, preds, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_valid, preds, alpha=0.3)
    plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()])
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title(title)
    plt.show()



def main():
    # Load data
    train_df = pd.read_csv('datasets/train.csv')
    test_df = pd.read_csv('datasets/test.csv')
    X, y, preprocessor = preprocess_data(train_df)
    X_train, X_valid, y_train, y_valid, num = split_data(X, y)
    

    # Train and evaluate the random forest and gradient boosting models
    rf_model, gb_model = train_model(X_train, y_train, preprocessor, num)
    rf_rmse = evaluate_model(rf_model, X_valid, y_valid)
    gb_rmse = evaluate_model(gb_model, X_valid, y_valid)
    
    print(f'Initial Random Forest RMSE: {rf_rmse}')
    print(f'Initial Gradient Boosting RMSE: {gb_rmse}')
    
    # Hyperparameter tuning 
    # CURRENT PARAMETERS SEEM TO RETURN BEST ACCURACY 
    param_grid_RF = {
        # 'model__n_estimators': [50, 100, 200, 300],
        # 'model__max_depth': [5, 10, 15, 20], 
        'model__n_estimators': [200],
        'model__max_depth': [10],

    }
    
    param_grid_GB = {
        # 'model__n_estimators': [100, 200, 300],
        # 'model__max_depth': [3, 5, 10], # 3 tends to be best
        'model__n_estimators': [300],
        'model__max_depth': [3],

    }
    
    best_params_rf, final_rf_model = hyperparameter_tuning(rf_model, param_grid_RF, X_train, y_train)
    best_params_gb, final_gb_model = hyperparameter_tuning(gb_model, param_grid_GB, X_train, y_train)
    
    print(f'Best parameters for RF: {best_params_rf}')
    print(f'Best parameters for GB: {best_params_gb}')
    
    # Final evaluation
    final_rf_rmse = evaluate_model(final_rf_model, X_valid, y_valid)
    final_gb_rmse = evaluate_model(final_gb_model, X_valid, y_valid)
    
    print(f'Final Random Forest RMSE: {final_rf_rmse}')
    print(f'Final Gradient Boosting RMSE: {final_gb_rmse}')
    
    # Make predictions on the test data and store them in csv files
    test_preds_rf = final_rf_model.predict(test_df)
    test_preds_gb = final_gb_model.predict(test_df)

    save_submission(test_df, test_preds_rf, 'predicted_results/rf_prices.csv')
    save_submission(test_df, test_preds_gb, 'predicted_results/gb_prices.csv')
    

    # After tuning graph results
    plot_results(y_valid, final_rf_model.predict(X_valid), 'Final Actual vs. Predicted Prices - Random Forest')
    # Before tuning
    plot_results(y_valid, rf_model.predict(X_valid), 'Initial Actual vs. Predicted Prices - Random Forest')

    plot_results(y_valid, final_gb_model.predict(X_valid), 'Final Actual vs. Predicted Prices - Gradient Boosting')
    plot_results(y_valid, gb_model.predict(X_valid), 'Initial Actual vs. Predicted Prices - Gradient Boosting')

if __name__ == '__main__':
    main()
