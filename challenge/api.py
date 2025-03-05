from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator, ValidationInfo
import pandas as pd
import xgboost as xgb

from challenge.model import DelayModel
from typing import List

app = FastAPI()
model = DelayModel()

# Define the input data
class PredictRequest(BaseModel):
    MES: int
    TIPOVUELO: str
    OPERA: str

    @field_validator('MES')
    def validate_month(cls, v, info: ValidationInfo):
        if v < 1 or v > 12:
            raise HTTPException(status_code=400, detail='MES is invalid')
        return v

    @field_validator('TIPOVUELO')
    def validate_flight_type(cls, v, info: ValidationInfo):
        valid_types = ['I', 'N']
        if v not in valid_types:
            raise HTTPException(status_code=400, detail='TIPOVUELO is invalid')
        return 

class PredictionInput(BaseModel):
    flights: List[PredictRequest]

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}
    
@app.post("/predict", status_code=200)
async def post_predict(requests: PredictionInput) -> dict:
    # Transform input data into the required format
    data = [
            {
            "OPERA": flight.OPERA,
            "MES": flight.MES,
            "TIPOVUELO": flight.TIPOVUELO
            } for flight in requests.flights
        ]
    df = pd.DataFrame(data)
    
    # Preprocess
    input_df = model.preprocess(df)
    
    # Make prediction
    prediction = model.predict(input_df)
    
    return {"predict": prediction}