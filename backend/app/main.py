"""
SurakshaNET FastAPI Backend
============================
Main application file with all API endpoints.

This backend accepts CSV files with network traffic data,
processes them through ML models, and returns predictions,
visualizations, and detailed reports.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import io
import logging
from pathlib import Path

from app.config import (
    API_TITLE, 
    API_VERSION, 
    API_DESCRIPTION, 
    MAX_FILE_SIZE, 
    UPLOADS_DIR
)
from app.models import HealthResponse, ReportResponse
from app.ml_service import ml_service
from app.visualization_service import viz_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION
)

# Configure CORS (allow frontend to communicate with backend)
# In production, replace "*" with specific frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend URL (e.g., "http://localhost:5173")
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount reports directory for serving HTML reports
project_root = Path(__file__).parent.parent.parent
reports_dir = project_root / "reports"
reports_dir.mkdir(exist_ok=True)

# Mount static files for reports
app.mount("/reports", StaticFiles(directory=str(reports_dir), html=True), name="reports")


@app.on_event("startup")
async def startup_event():
    """
    Run on application startup.
    Verify models are loaded and system is ready.
    """
    logger.info("="*70)
    logger.info("Starting SurakshaNET Backend API")
    logger.info("="*70)
    
    if ml_service.models_loaded:
        logger.info("✓ ML models loaded successfully")
    else:
        logger.error("✗ Failed to load ML models")
        logger.error("Please ensure models are trained and available in models/trained/")
    
    logger.info(f"✓ Upload directory: {UPLOADS_DIR}")
    logger.info("✓ API is ready to accept requests")
    logger.info("="*70)


@app.get("/", tags=["Health"])
async def root():
    """
    Root endpoint - basic health check.
    
    Returns:
        Simple message confirming API is running
    """
    return {
        "message": "SurakshaNET API is running",
        "version": API_VERSION,
        "status": "healthy"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Detailed health check endpoint.
    
    Returns:
        Detailed status including model loading status
    """
    return HealthResponse(
        status="healthy" if ml_service.models_loaded else "degraded",
        message="All systems operational" if ml_service.models_loaded else "Models not loaded",
        models_loaded=ml_service.get_model_status(),
        version=API_VERSION
    )


@app.post("/predict", response_model=ReportResponse, tags=["Prediction"])
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Main prediction endpoint.
    
    Accepts a CSV file with network traffic data and returns:
    - Predictions for each sample (binary and multi-class)
    - Summary statistics
    - Visualization plots (base64 encoded)
    
    Args:
        file: CSV file with network traffic features
        
    Returns:
        Complete analysis report with predictions and visualizations
        
    Raises:
        HTTPException: If file processing or prediction fails
    """
    logger.info(f"Received prediction request: {file.filename}")
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a CSV file."
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Check file size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.0f} MB"
            )
        
        # Parse CSV
        logger.info("Parsing CSV file...")
        df = pd.read_csv(io.BytesIO(content))
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Validate and prepare data
        X, feature_names = ml_service.validate_and_prepare_data(df)
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = ml_service.predict(X)
        
        # Format predictions (limit to first 1000 for API response)
        formatted_predictions = ml_service.format_predictions(predictions, limit=1000)
        
        # Calculate summary
        logger.info("Calculating summary statistics...")
        summary = ml_service.calculate_summary(predictions)
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        plots = viz_service.generate_all_plots(predictions)
        
        # Prepare response
        response = ReportResponse(
            summary=summary,
            predictions=formatted_predictions,
            plots=plots,
            message=f"Successfully analyzed {len(df)} network traffic samples"
        )
        
        logger.info("Prediction complete! Sending response...")
        return response
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Data validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)):
    """
    Batch prediction endpoint for large files.
    
    Similar to /predict but optimized for larger datasets.
    Returns only summary and limited sample predictions.
    
    Args:
        file: CSV file with network traffic features
        
    Returns:
        Summarized analysis with sample predictions
    """
    logger.info(f"Received batch prediction request: {file.filename}")
    
    # Similar to /predict but with optimizations
    # Process in chunks, return only summary
    # Implementation would be similar to /predict
    # but with streaming and chunk processing
    
    return {
        "message": "Batch prediction endpoint",
        "note": "Use /predict endpoint for now. Batch processing coming soon."
    }


@app.get("/models", tags=["Models"])
async def get_model_info():
    """
    Get information about loaded models.
    
    Returns:
        Details about available ML models
    """
    return {
        "models": ml_service.get_model_status(),
        "model_types": {
            "rf_binary": "Random Forest - Binary Classification",
            "rf_multiclass": "Random Forest - Multi-Class Classification",
            "xgb_binary": "XGBoost - Binary Classification",
            "xgb_multiclass": "XGBoost - Multi-Class Classification"
        },
        "primary_model": "xgb_binary",
        "classes": {
            "binary": ["Normal", "Attack"],
            "multiclass": ["Normal", "DoS", "Probe", "R2L", "U2R"]
        }
    }


@app.get("/stats", tags=["Statistics"])
async def get_stats():
    """
    Get general API statistics.
    
    Returns:
        API usage statistics (placeholder for now)
    """
    return {
        "message": "Statistics endpoint",
        "total_predictions": 0,  # Would track in database
        "uptime": "Running",
        "models_loaded": ml_service.models_loaded
    }


@app.post("/generate-report", tags=["Reports"])
async def generate_report():
    """
    Generate ML model evaluation report.
    
    Triggers the report generator to create a comprehensive
    HTML report with model metrics, confusion matrices, and plots.
    
    Returns:
        Report metadata and path
    """
    logger.info("Generating ML model evaluation report...")
    
    try:
        from pathlib import Path
        import sys
        
        # Add reporting module to path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        from backend.reporting.generate_report import ReportGenerator
        
        # Define paths
        data_path = project_root / "dataset" / "cleaned" / "nsl_kdd_cleaned.csv"
        model_dir = project_root / "backend" / "models" / "trained"
        report_dir = project_root / "reports"
        
        # Generate report
        generator = ReportGenerator(
            data_path=str(data_path),
            model_dir=str(model_dir),
            report_dir=str(report_dir)
        )
        
        report_path = generator.generate_report()
        
        return {
            "status": "success",
            "message": "Report generated successfully",
            "report_path": str(report_path),
            "report_name": report_path.name,
            "report_url": f"/reports/{report_path.name}"
        }
        
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.get("/reports", tags=["Reports"])
async def list_reports():
    """
    List all available reports.
    
    Returns:
        List of report files with metadata
    """
    try:
        from pathlib import Path
        from datetime import datetime
        
        project_root = Path(__file__).parent.parent.parent
        report_dir = project_root / "reports"
        
        if not report_dir.exists():
            return {"reports": []}
        
        reports = []
        for report_file in report_dir.glob("*.html"):
            stat = report_file.stat()
            reports.append({
                "name": report_file.name,
                "size": f"{stat.st_size / (1024*1024):.2f} MB",
                "created": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                "url": f"/reports/{report_file.name}"
            })
        
        # Sort by creation time (newest first)
        reports.sort(key=lambda x: x["created"], reverse=True)
        
        return {"reports": reports, "count": len(reports)}
        
    except Exception as e:
        logger.error(f"Failed to list reports: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    # Access at: http://localhost:8000
    # Docs at: http://localhost:8000/docs
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (development only)
        log_level="info"
    )
