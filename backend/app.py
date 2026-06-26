from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from contextlib import asynccontextmanager
from feature_engineer import AdvancedFeatureEngineer
import uvicorn

# Initialize FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield

app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_box_office_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
FEATURE_ENGINEER_PATH = os.path.join(MODEL_DIR, "feature_engineer.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

model = None
preprocessor = None
feature_engineer = None
feature_columns = []

# Data models
class MovieFeatures(BaseModel):
    budget: float = Field(..., gt=0, description="Budget in USD")
    popularity: float = Field(..., gt=0)
    runtime: float = Field(..., gt=0)
    release_year: int = Field(..., ge=1900, le=2100)
    release_month: int = Field(..., ge=1, le=12)
    release_day: int = Field(..., ge=1, le=31)
    original_language: str = Field(default="en")
    genre_count: int = Field(default=1, ge=1, le=5)
    companies_count: int = Field(default=1, ge=1, le=10)
    countries_count: int = Field(default=1, ge=1, le=5)
    languages_count: int = Field(default=1, ge=1, le=5)
    cast_count: int = Field(default=20, ge=1, le=200)
    crew_count: int = Field(default=50, ge=1, le=500)
    genres: List[str] = Field(default=["Drama"])

class PredictionRequest(BaseModel):
    movie: MovieFeatures

class PredictionResponse(BaseModel):
    predicted_revenue: float
    confidence_low: float
    confidence_high: float
    roi: float
    category: str
    message: str

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load artifacts
def load_artifacts():
    global model, preprocessor, feature_engineer, feature_columns
    try:
        # Make the class available in __main__ for pickle loading
        import __main__
        __main__.AdvancedFeatureEngineer = AdvancedFeatureEngineer
        
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        feature_engineer = joblib.load(FEATURE_ENGINEER_PATH)
        with open(FEATURE_COLUMNS_PATH, 'r') as f:
            feature_columns = json.load(f)
        print("Models loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# Feature alignment
def align_features(input_features, selected_genres):
    aligned_df = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    
    # Map basic features
    mapping = {
        'budget': 'budget', 'popularity': 'popularity', 'runtime': 'runtime',
        'release_year': 'release_year', 'release_month': 'release_month',
        'release_day': 'release_day', 'genre_count': 'genre_count',
        'companies_count': 'companies_count', 'countries_count': 'countries_count',
        'languages_count': 'languages_count', 'cast_count': 'cast_count',
        'crew_count': 'crew_count'
    }
    
    for input_key, df_key in mapping.items():
        if input_key in input_features and df_key in aligned_df.columns:
            aligned_df[df_key] = float(input_features[input_key])
    
    # Add genres
    if feature_engineer and hasattr(feature_engineer, 'top_genres'):
        for genre in feature_engineer.top_genres:
            safe_genre = genre.replace(" ", "_").replace("&", "and")
            genre_col = f'genre_{safe_genre}'
            if genre_col in aligned_df.columns:
                aligned_df[genre_col] = 1 if genre in selected_genres else 0
    
    # Add derived features
    if 'budget' in input_features and 'runtime' in input_features:
        budget = input_features['budget']
        runtime = input_features['runtime']
        if 'budget_per_minute' in aligned_df.columns:
            aligned_df['budget_per_minute'] = budget / (runtime + 1)
        if 'log_budget' in aligned_df.columns:
            aligned_df['log_budget'] = np.log1p(budget)
    
    if 'popularity' in input_features and 'log_popularity' in aligned_df.columns:
        aligned_df['log_popularity'] = np.log1p(input_features['popularity'])
    
    return aligned_df

# Categorize revenue
def categorize_revenue(revenue):
    if revenue > 500000000:
        return "BLOCKBUSTER"
    elif revenue > 200000000:
        return "HIT"
    elif revenue > 50000000:
        return "MODERATE"
    else:
        return "RISKY"

# Load on startup
# Lifespan handles startup

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("../frontend/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content, headers={"Cache-Control": "no-cache"})
    except FileNotFoundError:
        return {"message": "Box Office Prediction API", "status": "active"}

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model else "degraded",
        "model_loaded": model is not None,
        "feature_count": len(feature_columns)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    
    movie_dict = request.movie.dict()
    genres = movie_dict.pop("genres", [])
    
    # Align features
    aligned_df = align_features(movie_dict, genres)
    aligned_processed = preprocessor.transform(aligned_df)
    
    # Predict
    prediction = model.predict(aligned_processed)[0]
    
    # Calculate confidence
    if prediction > 500000000:
        conf = 0.15
    elif prediction > 100000000:
        conf = 0.20
    elif prediction > 50000000:
        conf = 0.25
    else:
        conf = 0.35
    
    confidence_low = prediction * (1 - conf)
    confidence_high = prediction * (1 + conf)
    
    # ROI
    budget = movie_dict.get("budget", 1)
    roi = prediction / budget if budget > 0 else 0
    
    # Category
    category = categorize_revenue(prediction)
    
    messages = {
        "BLOCKBUSTER": "Excellent investment! High success probability.",
        "HIT": "Good investment potential. Likely profitable.",
        "MODERATE": "Moderate risk. Could break even.",
        "RISKY": "High risk. Consider budget reduction."
    }
    
    return PredictionResponse(
        predicted_revenue=float(prediction),
        confidence_low=float(confidence_low),
        confidence_high=float(confidence_high),
        roi=float(roi),
        category=category,
        message=messages.get(category, "Prediction complete.")
    )

# Frontend interface
@app.get("/predict-form")
async def predict_form():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Box Office Predictor</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
            h1 { color: #333; text-align: center; }
            .form-group { margin: 15px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
            button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #45a049; }
            .result { margin-top: 20px; padding: 15px; background: white; border-radius: 5px; display: none; }
            .result.show { display: block; }
            .revenue { font-size: 2em; font-weight: bold; color: #27ae60; text-align: center; }
            .loading { text-align: center; padding: 20px; display: none; }
            .loading.show { display: block; }
            .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Box Office Revenue Predictor</h1>
            
            <form id="movieForm">
                <div class="form-group">
                    <label>Budget (in millions USD)</label>
                    <input type="number" id="budget" value="150" step="0.1" required>
                </div>
                
                <div class="form-group">
                    <label>Popularity Score</label>
                    <input type="number" id="popularity" value="250.5" step="0.1" required>
                </div>
                
                <div class="form-group">
                    <label>Runtime (minutes)</label>
                    <input type="number" id="runtime" value="135" required>
                </div>
                
                <div class="form-group">
                    <label>Release Date</label>
                    <input type="date" id="releaseDate" value="2024-06-15" required>
                </div>
                
                <div class="form-group">
                    <label>Genres (comma separated)</label>
                    <input type="text" id="genres" value="Action, Adventure, Sci-Fi" required>
                </div>
                
                <button type="button" onclick="predict()">Predict Revenue</button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Predicting...</p>
            </div>
            
            <div class="result" id="result">
                <h2>Prediction Results</h2>
                <div class="revenue" id="revenue">$0</div>
                <p><strong>Category:</strong> <span id="category"></span></p>
                <p><strong>ROI:</strong> <span id="roi"></span></p>
                <p><strong>Confidence Range:</strong> <span id="confidence"></span></p>
                <p id="message"></p>
            </div>
        </div>
        
        <script>
            async function predict() {
                // Show loading
                document.getElementById('loading').classList.add('show');
                document.getElementById('result').classList.remove('show');
                
                try {
                    // Parse date
                    const date = document.getElementById('releaseDate').value;
                    const [year, month, day] = date.split('-');
                    
                    // Parse genres
                    const genres = document.getElementById('genres').value
                        .split(',')
                        .map(g => g.trim())
                        .filter(g => g);
                    
                    // Prepare request
                    const data = {
                        movie: {
                            budget: parseFloat(document.getElementById('budget').value) * 1000000,
                            popularity: parseFloat(document.getElementById('popularity').value),
                            runtime: parseFloat(document.getElementById('runtime').value),
                            release_year: parseInt(year),
                            release_month: parseInt(month),
                            release_day: parseInt(day),
                            original_language: "en",
                            genre_count: genres.length,
                            companies_count: 5,
                            countries_count: 2,
                            languages_count: 2,
                            cast_count: 45,
                            crew_count: 120,
                            genres: genres
                        }
                    };
                    
                    // Make API call
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    
                    if (!response.ok) throw new Error('Prediction failed');
                    
                    const result = await response.json();
                    
                    // Display results
                    document.getElementById('revenue').textContent = 
                        formatCurrency(result.predicted_revenue);
                    document.getElementById('category').textContent = result.category;
                    document.getElementById('roi').textContent = result.roi.toFixed(2) + 'x';
                    document.getElementById('confidence').textContent = 
                        formatCurrency(result.confidence_low) + ' - ' + formatCurrency(result.confidence_high);
                    document.getElementById('message').textContent = result.message;
                    
                    document.getElementById('result').classList.add('show');
                    
                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    document.getElementById('loading').classList.remove('show');
                }
            }
            
            function formatCurrency(amount) {
                if (amount >= 1000000000) return '$' + (amount/1000000000).toFixed(2) + 'B';
                if (amount >= 1000000) return '$' + (amount/1000000).toFixed(2) + 'M';
                if (amount >= 1000) return '$' + (amount/1000).toFixed(2) + 'K';
                return '$' + amount.toFixed(2);
            }
            
            // Set today's date as default
            window.onload = function() {
                const today = new Date().toISOString().split('T')[0];
                document.getElementById('releaseDate').value = today;
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)