from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
from .model_registry import retrieve

app = FastAPI()

model, features = retrieve('used_car_price_prediction')

@app.get("/")
def home():
    return {"message": "Welcome to DSSI!"}

@app.post("/api/v1/usedcar/predict")
def predict(data: dict):
    pred_df = pd.DataFrame.from_dict([data])
    pred = model.predict(pred_df[features])
    return {"price": pred[0]}
