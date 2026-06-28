import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import io

app = FastAPI(
    title="California House Price Prediction API",
    description="Predicts California house prices using a trained Random Forest Regression model.",
    version="1.0"
)

# Load trained model and feature names
model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")


# Input schema
class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Median Income of Neighbourhood")
    HouseAge: float = Field(gt=0, description="Average age of houses in the block")
    AveRooms: float = Field(gt=0, description="Average number of rooms per house in the block")
    AveBedrms: float = Field(gt=0, description="Average number of bedrooms per house in the block")
    Population: float = Field(gt=0, description="Total population of the block")
    AveOccup: float = Field(gt=0, description="Average occupancy per household")
    Latitude: float = Field(ge=32, le=42, description="Latitude")
    Longitude: float = Field(ge=-125, le=-114, description="Longitude")


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "California House Price Prediction API",
        "status": "running",
        "endpoint": "Send POST request to /predict"
    }


# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "RandomForestRegressor",
        "features": features,
        "avg_error": "$39,000"
    }


# Single house prediction endpoint
@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        input_data = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AveRooms,
            "AveBedrms": house.AveBedrms,
            "Population": house.Population,
            "AveOccup": house.AveOccup,
            "Latitude": house.Latitude,
            "Longitude": house.Longitude
        }])

        predicted = model.predict(input_data)[0]
        price_usd = predicted * 100000

        return {
            "predicted_price": f"${price_usd:,.0f}",
            "predicted_price_short": f"${predicted:.2f} x $100,000",
            "confidence_range": f"${price_usd - 39000:,.0f} to ${price_usd + 39000:,.0f}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# CSV file prediction endpoint
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):

    # Check file extension
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file only."
        )

    # Read uploaded file
    contents = await file.read()

    # Convert CSV into DataFrame
    df = pd.read_csv(io.BytesIO(contents))

    # Remove extra spaces from column names (if any)
    df.columns = df.columns.str.strip()

    # Required feature columns
    required_columns = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude"
    ]

    # Check for missing columns
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Missing columns: {missing_columns}"
        )

    # Predict prices
    predictions = model.predict(df[required_columns])

    # Add prediction column
    df["PredictedPrice"] = predictions * 100000

    # Format prices as currency
    df["PredictedPrice"] = df["PredictedPrice"].apply(
        lambda x: f"${x:,.0f}"
    )

    # Convert DataFrame back to CSV
    output = df.to_csv(index=False)

    # Return downloadable CSV
    return StreamingResponse(
        io.StringIO(output),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=predictions.csv"
        }
    )