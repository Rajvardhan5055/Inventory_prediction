from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # <-- 1. We imported this
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Smart Retail Inventory API", description="AI Demand Forecaster")

# --- 2. THIS IS THE NEW SECURITY UPDATE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # The "*" means "allow any web page to connect"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------------------------------------

print("Loading the Random Forest Model...")
try:
    model = joblib.load('models/rf_model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

class SalesRequest(BaseModel):
    store: int
    item: int
    year: int
    month: int
    day: int
    dayofweek: int
    is_weekend: int

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
