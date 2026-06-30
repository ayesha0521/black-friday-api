
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

app = FastAPI()

class PurchaseInput(BaseModel):
    Gender: str
    Age: str
    Occupation: int
    City_Category: str
    Stay_In_Current_City_Years: str
    Marital_Status: int
    Product_Category_1: int
    Product_Category_2: float
    Product_Category_3: float

@app.get("/")
def home():
    return {"message": "API is running!"}

@app.post("/predict")
def predict(data: PurchaseInput):
    input_dict = dict(data)
    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df, columns=["Gender", "Age", "City_Category",
                                                   "Stay_In_Current_City_Years"])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(input_df)[0]
    return {
        "predicted_purchase": round(float(prediction), 2),
        "model_used": str(type(model).__name__)
    }
