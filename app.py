from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("best_house_price_model.pkl")

# Full features list (all columns required by the trained model)
training_columns = [
    "LotArea", "YearBuilt", "FullBath", "GarageCars", "OverallQual",
    "GarageType", "BsmtFinSF1", "LotFrontage", "OverallCond", "Street",
    "MSSubClass", "Neighborhood", "ExterCond", "BsmtFullBath", "PoolQC",
    "LandSlope", "RoofMatl", "PoolArea", "OpenPorchSF", "BsmtUnfSF",
    "TotalBsmtSF", "GarageFinish", "BsmtHalfBath", "Foundation", "Heating",
    "Utilities", "LowQualFinSF", "WoodDeckSF", "MasVnrType", "Functional",
    "SaleType", "HouseStyle", "Condition2", "BsmtCond", "BsmtFinType1",
    "BsmtFinSF2", "Condition1", "SaleCondition", "Fence", "YrSold",
    "HalfBath", "GarageArea", "BsmtFinType2", "LandContour", "LotShape",
    "FireplaceQu", "EnclosedPorch", "HeatingQC", "GrLivArea", "3SsnPorch",
    "BldgType", "MoSold", "BedroomAbvGr", "YearRemodAdd", "Exterior2nd",
    "TotRmsAbvGrd", "KitchenAbvGr", "MasVnrArea", "GarageYrBlt",
    "GarageCond", "LotConfig", "Exterior1st", "1stFlrSF", "MiscFeature",
    "GarageQual", "MiscVal", "PavedDrive", "2ndFlrSF", "KitchenQual",
    "CentralAir", "Alley", "BsmtQual", "ExterQual", "MSZoning",
    "Fireplaces", "BsmtExposure", "Electrical", "RoofStyle", "ScreenPorch"
]

@app.route("/")
def home():
    return jsonify({"message": "House Price Prediction API Running..."})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([data])

        # fill missing numerical columns with default 0
        for col in training_columns:
            if col not in input_df.columns:
                if col in ["GarageCars", "BsmtFullBath", "FullBath", "HalfBath", "YrSold"]:  
                    input_df[col] = 0  
                else:
                    input_df[col] = np.nan

        input_df = input_df[training_columns]

        # make the prediction
        prediction = model.predict(input_df)[0]
        predicted_price = np.expm1(prediction)

        return jsonify(
            {"PredictedSalePrice": round(predicted_price, 2)}
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
