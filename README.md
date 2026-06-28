# California House Price Prediction API

## About the Project

This project is a Machine Learning API developed using **FastAPI** and **Scikit-learn**. It predicts the price of houses in California based on different housing features such as median income, house age, average number of rooms, population, latitude, and longitude.

The model was trained using the California Housing dataset available in Scikit-learn and deployed as a REST API using FastAPI.

Apart from predicting the price of a single house, the API also supports uploading a CSV file to predict prices for multiple houses at once.

---

## Features

- Predict house prices using Machine Learning
- Single house price prediction
- Batch prediction using CSV upload
- Download predictions as a CSV file
- Input validation using Pydantic
- Automatic API documentation using Swagger UI
- Health check endpoint

---

## Technologies Used

- Python
- FastAPI
- Scikit-learn
- Pandas
- Joblib
- Pydantic
- Uvicorn

---

## Project Structure

```
FastAPIProject2/
│
├── explore.py
├── train.py
├── main.py
├── house_model.joblib
├── house_features.joblib
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

The project uses the California Housing dataset provided by Scikit-learn.

### Input Features

- MedInc
- HouseAge
- AveRooms
- AveBedrms
- Population
- AveOccup
- Latitude
- Longitude

### Target

- House Price

---

## Machine Learning Model

The model used in this project is **Random Forest Regressor**.

The dataset was divided into training and testing sets using an 80:20 split.

The trained model achieved an average prediction error (Mean Absolute Error) of approximately **$39,000**.

---

## API Endpoints

### Home

```
GET /
```

Returns basic information about the API.

---

### Health Check

```
GET /health
```

Displays the API status, model information, feature names, and average prediction error.

---

### Predict House Price

```
POST /predict
```

Example Request

```json
{
  "MedInc": 8.3252,
  "HouseAge": 41,
  "AveRooms": 6.984,
  "AveBedrms": 1.023,
  "Population": 322,
  "AveOccup": 2.555,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

Example Response

```json
{
  "predicted_price": "$426,579",
  "predicted_price_short": "$4.27 x $100,000",
  "confidence_range": "$387,579 to $465,579"
}
```

---

### Predict Using CSV File

```
POST /predict-file
```

Upload a CSV file containing the required feature columns.

The API returns a downloadable CSV file with an additional `PredictedPrice` column.

---

## Running the Project

### Clone the repository

```bash
git clone https://github.com/your-username/California-House-Price-Prediction-API.git
```

### Move to the project folder

```bash
cd California-House-Price-Prediction-API
```

### Install the required packages

```bash
pip install -r requirements.txt
```

### Start the FastAPI server

```bash
uvicorn main:app --reload
```

Open your browser and visit

```
http://127.0.0.1:8000/docs
```

to access the Swagger UI.

---

## Future Improvements

Some improvements that can be made in the future include:

- Deploying the API on a cloud platform
- Using more advanced regression models
- Adding user authentication
- Creating a web interface for predictions

---

## Author

**Rida Shaikh**

Bachelor of Computer Applications (BCA)

This project was developed as part of my Machine Learning and FastAPI learning journey.
To understand the complete machine learning workflow, from data exploration and model training to deploying a prediction API using FastAPI