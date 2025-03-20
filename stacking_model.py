import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# HalvingGridSearchCV if needed
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

RANDOM_STATE = 42  

def preprocess_data(df):
    """handles missing values, scales numerical features, and encodes categorical ones."""
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    df["SalePrice"] = np.log1p(df["SalePrice"])

    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    numerical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ], sparse_threshold=0.0)

    return X, y, preprocessor


def split_data(X, y):
    return train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=RANDOM_STATE)

def train_model(X_train, y_train, preprocessor, model_type="rf", progress_bar=None):
    """trains a regression model with improved parameters."""
    models = {
        "rf": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_STATE),
        "gb": GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=RANDOM_STATE),
        "xgb": xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE),
        "lgb": lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, min_data_in_leaf=20, random_state=RANDOM_STATE, verbose=-1),
        "histgb": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE)
    }

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", models[model_type])
    ])

    if progress_bar:
        progress_bar.set_description(f"Training {model_type.upper()}...")
        sleep(0.5)

    pipeline.fit(X_train, y_train)

    if progress_bar:
        progress_bar.update(1)

    return pipeline

def train_stacking_model(X_train, y_train, preprocessor, progress_bar=None):
    """trains a Stacking Regressor using RF, GB, XGBoost, and LightGBM."""
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=200, random_state=42)),
        ('xgb', xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)),
        ('lgb', lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, min_data_in_leaf=20, random_state=42, verbose=-1))
    ]

    stack_model = StackingRegressor(estimators=base_models, final_estimator=Ridge())

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', stack_model)
    ])

    if progress_bar:
        progress_bar.set_description("Training Stacking Model...")
        sleep(0.5)

    pipeline.fit(X_train, y_train)

    if progress_bar:
        progress_bar.update(1)

    return pipeline

def evaluate_model(model, X_valid, y_valid):
    """calculates the model using RMSE and R^2"""
    preds = model.predict(X_valid)
    rmse = root_mean_squared_error(y_valid, preds)
    r2 = r2_score(y_valid, preds)
    return rmse, r2

def save_results(results):
    """Saves model results to CSV file"""
    results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R²"])
    results_df.to_csv("model_results.csv", index=False)
    print("Model results saved to model_results.csv!")

def save_predictions(test_df, predictions, filename):
    """Saves test predictions to CSV file"""
    submission = pd.DataFrame({"Id": test_df.index, "SalePrice": np.expm1(predictions)})  
    submission.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}!")

def plot_results(y_valid, preds, title):
    """Plots actual vs predicted prices"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_valid, preds, alpha=0.3, label="Predicted")
    plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()])
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title(title)
    plt.legend()
    plt.savefig(f"graphs/{title}.png")  
    plt.close()  



def main():
    print("Loading data...")
    train_df = pd.read_csv("datasets/train.csv")
    test_df = pd.read_csv("datasets/test.csv")

    print("Preprocessing data...")
    X, y, preprocessor = preprocess_data(train_df)
    X_train, X_valid, y_train, y_valid = split_data(X, y)

    model_types = ["rf", "gb", "xgb", "lgb", "histgb", "stacking"]

    print("Training models with progress bar...")
    with tqdm(total=len(model_types), desc="Training Models") as pbar:
        models = {
            model_type: train_model(X_train, y_train, preprocessor, model_type, pbar) if model_type != "stacking"
            else train_stacking_model(X_train, y_train, preprocessor, pbar)
            for model_type in model_types
        }

    print("Evaluating models...")
    results = [(name.upper(), *evaluate_model(model, X_valid, y_valid)) for name, model in models.items()]
    save_results(results)

    # Best RMSE from models 
    best_model = min(results, key=lambda x: x[1])  
    best_model_name = best_model[0].lower()

    print(f"Best model: {best_model_name.upper()} (RMSE: {best_model[1]:.4f})")
    joblib.dump(models[best_model_name], "best_house_price_model.pkl")

    # Generate predictions with the best model
    test_preds = models[best_model_name].predict(test_df)
    save_predictions(test_df, test_preds, f"predicted_results/{best_model_name}.csv")

    plot_results(y_valid, models[best_model_name].predict(X_valid), f"Best Model ({best_model_name.upper()})")

if __name__ == "__main__":
    main()
