"""
FastAPI Application for ML Model Serving - PROPERLY FIXED VERSION
Handles predictions with correct feature order
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import time
from functools import lru_cache
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Model API",
    description="Production-ready ML model serving with FastAPI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
preprocessor_data = None
prediction_count = 0
start_time = datetime.now()

# ==================== Pydantic Models ====================

class Features(BaseModel):
    """Input features for prediction"""
    age: int = Field(..., ge=18, le=100)
    tenure: int = Field(..., ge=0, le=72)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    contract_type: str
    payment_method: str
    internet_service: str
    online_security: str
    tech_support: str
    streaming_tv: str

class PredictionRequest(BaseModel):
    """Single prediction request"""
    features: Features

class PredictionResponse(BaseModel):
    """Prediction response"""
    prediction: int
    probability: float
    confidence: str
    timestamp: str
    processing_time_ms: Optional[float] = None

# ==================== Helper Functions ====================

@lru_cache()
def get_model():
    """Load model and preprocessor"""
    global model, preprocessor_data
    try:
        logger.info("Loading model...")
        model = joblib.load('models/best_model.pkl')
        logger.info("Model loaded successfully")
        
        # Load preprocessor
        try:
            with open('data/processed/preprocessor.pkl', 'rb') as f:
                preprocessor_data = pickle.load(f)
            logger.info(f"Preprocessor loaded - {len(preprocessor_data['feature_names'])} features")
        except Exception as e:
            logger.warning(f"Could not load preprocessor: {e}")
            preprocessor_data = None
            
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed")

def preprocess_features(features: Features) -> pd.DataFrame:
    """Preprocess features matching training pipeline exactly"""
    
    # Step 1: Create raw features
    df = pd.DataFrame([{
        'age': features.age,
        'tenure': features.tenure,
        'monthly_charges': features.monthly_charges,
        'total_charges': features.total_charges,
        'contract_type': features.contract_type,
        'payment_method': features.payment_method,
        'internet_service': features.internet_service,
        'online_security': features.online_security,
        'tech_support': features.tech_support,
        'streaming_tv': features.streaming_tv,
    }])
    
    # Step 2: One-hot encode categorical variables
    # Contract type
    for val in ['Month-to-month', 'One year', 'Two year']:
        df[f'contract_type_{val}'] = (df['contract_type'] == val).astype(int)
    df = df.drop('contract_type', axis=1)
    
    # Payment method
    for val in ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']:
        df[f'payment_method_{val}'] = (df['payment_method'] == val).astype(int)
    df = df.drop('payment_method', axis=1)
    
    # Internet service
    for val in ['DSL', 'Fiber optic', 'No']:
        df[f'internet_service_{val}'] = (df['internet_service'] == val).astype(int)
    df = df.drop('internet_service', axis=1)
    
    # Convert online_security, tech_support, streaming_tv to binary
    # These are "Yes" or "No" strings, convert to 1 or 0
    df['online_security'] = (df['online_security'] == 'Yes').astype(int)
    df['tech_support'] = (df['tech_support'] == 'Yes').astype(int)
    df['streaming_tv'] = (df['streaming_tv'] == 'Yes').astype(int)
    
    # Step 3: Engineer features
    # total_value
    df['total_value'] = df['tenure'] * df['monthly_charges']
    
    # tenure_group
    tenure_val = df['tenure'].iloc[0]
    df['tenure_group_0-1yr'] = 1 if 0 <= tenure_val < 12 else 0
    df['tenure_group_1-2yr'] = 1 if 12 <= tenure_val < 24 else 0
    df['tenure_group_2-4yr'] = 1 if 24 <= tenure_val < 48 else 0
    df['tenure_group_4-6yr'] = 1 if 48 <= tenure_val <= 72 else 0
    
    # Step 4: Reorder columns to match training
    if preprocessor_data and 'feature_names' in preprocessor_data:
        feature_names = preprocessor_data['feature_names']
        # Add missing columns
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        # Reorder
        df = df[feature_names]
    
    # Step 5: Scale features
    if preprocessor_data and 'scaler' in preprocessor_data:
        scaler = preprocessor_data['scaler']
        df_scaled = pd.DataFrame(
            scaler.transform(df),
            columns=df.columns
        )
        return df_scaled
    
    return df

def get_confidence(probability: float) -> str:
    """Determine confidence level"""
    if probability >= 0.8 or probability <= 0.2:
        return "high"
    elif probability >= 0.6 or probability <= 0.4:
        return "medium"
    else:
        return "low"

# ==================== API Endpoints ====================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting ML API...")
    get_model()
    logger.info("API ready to serve predictions")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "ML Model API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Make predictions",
            "/docs": "API documentation"
        }
    }

@app.get("/health", tags=["Health"])
async def health():
    """Health check endpoint"""
    uptime = (datetime.now() - start_time).total_seconds()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor_data is not None,
        "uptime_seconds": uptime,
        "total_predictions": prediction_count
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest, model=Depends(get_model)):
    """Make single prediction"""
    global prediction_count
    
    start = time.time()
    
    try:
        # Preprocess features
        X = preprocess_features(request.features)
        logger.info(f"Preprocessed shape: {X.shape}, columns: {list(X.columns)[:5]}...")
        
        # Make prediction
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])
        
        # Get confidence level
        confidence = get_confidence(probability)
        
        # Update metrics
        prediction_count += 1
        
        processing_time = (time.time() - start) * 1000
        
        logger.info(f"Prediction: {prediction} (prob: {probability:.3f})")
        
        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Error Handlers ====================

from starlette.requests import Request
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail), "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)