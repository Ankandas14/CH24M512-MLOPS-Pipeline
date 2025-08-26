from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
from prediction import ModelPredictor

app = FastAPI()
predictor = ModelPredictor()

class TitanicInput(BaseModel):
    # Define all input fields required for prediction
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    # Add other fields as needed

@app.post("/predict")
async def predict(input_data: TitanicInput):
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])
    prediction = predictor.predict(input_df)
    return {"prediction": prediction}

@app.on_event("shutdown")
def shutdown_event():
    predictor.stop()
