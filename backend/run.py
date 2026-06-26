#!/usr/bin/env python3
"""
Simple run script for the Box Office Prediction API
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if all dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import joblib
        import pandas
        import numpy
        import sklearn
        import xgboost
        print("All dependencies are installed ✓")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("\nPlease install all dependencies:")
        print("pip install -r requirements.txt")
        return False

def check_model_files():
    """Check if model files exist"""
    model_files = [
        "models/best_box_office_model.pkl",
        "models/preprocessor.pkl", 
        "models/feature_engineer.pkl",
        "models/feature_columns.json"
    ]
    
    missing = []
    for file in model_files:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        print("\n⚠️  Missing model files:")
        for file in missing:
            print(f"   - {file}")
        print("\nPlease make sure you have:")
        print("1. Trained the model in the Jupyter notebook")
        print("2. Copied the model files to backend/models/")
        return False
    
    print("All model files found ✓")
    return True

def main():
    """Main function to run the API"""
    print("=" * 60)
    print("BOX OFFICE PREDICTION SYSTEM")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check model files
    if not check_model_files():
        response = input("\nContinue anyway? (y/n): ").lower().strip()
        if response != 'y':
            return 1
    
    print("\n" + "=" * 60)
    print("STARTING SERVER...")
    print("=" * 60)
    print("\nAPI Endpoints:")
    print("  • http://localhost:8000          - API Documentation")
    print("  • http://localhost:8000/docs     - Swagger UI")
    print("  • http://localhost:8000/predict-form - Prediction Form")
    print("\nTo use the frontend:")
    print("  Open frontend/index.html in your browser")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Run the FastAPI app
        import uvicorn
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())