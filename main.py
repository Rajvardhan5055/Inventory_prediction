from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field # <-- 1. Added 'Field' here
import joblib
import pandas as pd

app = FastAPI(title="Enterprise AI Forecaster")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading the Random Forest Model...")
try:
    model = joblib.load('models/rf_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")

# --- THIS IS THE NEW VALIDATION Bouncer ---
class SalesRequest(BaseModel):
    store: int = Field(..., ge=1, le=10, description="Store ID must be between 1 and 10")
    item: int = Field(..., ge=1, le=50, description="Item ID must be between 1 and 50")
    year: int = Field(..., ge=2013, le=2030)
    month: int = Field(..., ge=1, le=12)
    day: int = Field(..., ge=1, le=31)
    dayofweek: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)
# ------------------------------------------

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.post("/predict")
def predict_sales(request: SalesRequest):
    input_data = pd.DataFrame([{
        'store': request.store,
        'item': request.item,
        'year': request.year,
        'month': request.month,
        'day': request.day,
        'dayofweek': request.dayofweek,
        'is_weekend': request.is_weekend
    }])
    prediction = model.predict(input_data)
    return {
        "store_id": request.store,
        "item_id": request.item,
        "predicted_sales": round(prediction[0], 2),
        "status": "Success"
    }
