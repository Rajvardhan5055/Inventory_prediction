from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
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

class SalesRequest(BaseModel):
    store: int
    item: int
    year: int
    month: int
    day: int
    dayofweek: int
    is_weekend: int

# --- THIS IS THE NEW PART: Serving your Website ---
@app.get("/")
def serve_frontend():
    return FileResponse("index.html")
# --------------------------------------------------

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
