"""
SurakshaNET FastAPI Backend
============================
Main application file with all API endpoints.

This backend accepts CSV files with network traffic data,
processes them through ML models, and returns predictions,
visualizations, and detailed reports.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import pandas as pd
import io
import logging
import asyncio
from pathlib import Path
from datetime import datetime

from app.config import (
    API_TITLE, 
    API_VERSION, 
    API_DESCRIPTION, 
    MAX_FILE_SIZE, 
    UPLOADS_DIR,
    MAX_ROWS_FOR_VISUALIZATION,
    PREDICTION_BATCH_SIZE,
    MULTICLASS_LABELS
)
from app.models import HealthResponse, ReportResponse
from app.ml_service import ml_service
from app.visualization_service import viz_service
from app.realtime_service import initialize_realtime_service
import app.realtime_service  # Import module to access global later
from app.realtime_viz_service import realtime_viz_service

# Import network logger and tester
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from network_logger.network_logger import NetworkLogger
from network_tester.network_tester import NetworkTester, TrafficType
from blocking.ipblocking import ip_blocking_service

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

# IP Blocking Middleware
@app.middleware("http")
async def block_malicious_ips(request: Request, call_next):
    """Check if IP is blocked before processing request"""
    client_ip = request.client.host
    
    if ip_blocking_service.is_blocked(client_ip):
        logger.warning(f"Blocked request from: {client_ip}")
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: IP blocked due to suspicious activity"
        )
    
    response = await call_next(request)
    return response

# Mount reports directory for serving HTML reports
project_root = Path(__file__).parent.parent.parent
reports_dir = project_root / "reports"
reports_dir.mkdir(exist_ok=True)

# Mount static files for reports
app.mount("/reports", StaticFiles(directory=str(reports_dir), html=True), name="reports")

# Initialize network logger
project_root_path = Path(__file__).parent.parent.parent
logs_dir = project_root_path / "backend" / "logs"
logs_dir.mkdir(exist_ok=True)
network_logger = NetworkLogger(log_file=str(logs_dir / "network_logs.json"))

# Initialize real-time monitoring service IMMEDIATELY (not in startup event)
initialize_realtime_service(network_logger)
logger.info("Real-time monitoring service initialized at module load")


@app.on_event("startup")
async def startup_event():
    """
    Run on application startup.
    Verify models are loaded and system is ready.
    """
    import app.realtime_service as rt_module
    
    logger.info("="*70)
    logger.info("Starting SurakshaNET Backend API")
    logger.info("="*70)
    
    # Verify real-time service is initialized
    if rt_module.realtime_service is not None:
        logger.info("✓ Real-time monitoring service ready")
    else:
        logger.error("✗ Real-time service failed to initialize")
    
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
            logger.warning(f"File too large: {len(content) / (1024*1024):.2f} MB")
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({len(content) / (1024*1024):.1f} MB). Maximum size is {MAX_FILE_SIZE / (1024*1024):.0f} MB"
            )
        
        # Parse CSV
        logger.info("Parsing CSV file...")
        df = pd.read_csv(io.BytesIO(content))
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Check for empty dataframe
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="CSV file contains no data rows")
        
        # Warn about large datasets
        if len(df) > MAX_ROWS_FOR_VISUALIZATION:
            logger.warning(f"Large dataset ({len(df)} rows). Visualizations will be limited.")
        
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
        
        # Generate visualizations (skip for very large datasets to save memory)
        logger.info("Generating visualizations...")
        if len(df) <= MAX_ROWS_FOR_VISUALIZATION:
            plots = viz_service.generate_all_plots(predictions)
        else:
            logger.warning(f"Dataset too large ({len(df)} rows) for full visualizations. Generating summary plots only.")
            # Generate only essential plots for large datasets
            plots = viz_service.generate_summary_plots(predictions)
        
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
        logger.error("CSV file is empty")
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError as e:
        logger.error(f"Invalid CSV format: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
    except ValueError as e:
        logger.error(f"Data validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Data validation error: {str(e)}")
    except HTTPException:
        # Re-raise HTTPException as-is (already has correct status code)
        raise
    except Exception as e:
        logger.error(f"Unexpected prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction. Please check logs.")


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
        },
        "features": {
            "expected": ml_service.expected_features or "flexible",
            "adaptation_enabled": True,
            "supported_range": "3+ features"
        },
        "capabilities": [
            "Flexible feature count handling",
            "Automatic feature engineering",
            "Binary and multi-class classification",
            "Real-time intrusion detection"
        ]
    }


@app.post("/validate-dataset", tags=["Validation"])
async def validate_dataset(file: UploadFile = File(...)):
    """
    Validate if a CSV file is compatible with the ML models.
    
    This endpoint checks the file without making predictions,
    useful for verifying dataset format before processing.
    
    Args:
        file: CSV file to validate
        
    Returns:
        Validation results with detailed feedback
    """
    logger.info(f"Validating dataset: {file.filename}")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a CSV file."
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Check file size
        file_size_mb = len(content) / (1024*1024)
        if len(content) > MAX_FILE_SIZE:
            return {
                "valid": False,
                "error": f"File too large ({file_size_mb:.1f} MB). Maximum size is {MAX_FILE_SIZE / (1024*1024):.0f} MB",
                "file_size_mb": round(file_size_mb, 2),
                "suggestion": "Try uploading a smaller file or use a sample of your dataset"
            }
        
        # Parse CSV
        df = pd.read_csv(io.BytesIO(content))
        
        # Check if empty
        if len(df) == 0:
            return {
                "valid": False,
                "error": "CSV file is empty",
                "rows": 0,
                "columns": 0
            }
        
        # Try to validate and prepare data
        try:
            X, feature_names = ml_service.validate_and_prepare_data(df)
            
            return {
                "valid": True,
                "message": "✅ Dataset is compatible with the model",
                "file_size_mb": round(file_size_mb, 2),
                "rows": len(df),
                "columns": len(df.columns),
                "features": len(feature_names),
                "expected_features": 41,
                "sample_features": feature_names[:5] if len(feature_names) > 5 else feature_names,
                "ready_for_prediction": True
            }
            
        except ValueError as e:
            return {
                "valid": False,
                "error": str(e),
                "file_size_mb": round(file_size_mb, 2),
                "rows": len(df),
                "columns": len(df.columns),
                "features_found": len(df.columns) - 1 if 'label' in df.columns else len(df.columns),
                "expected_features": 41,
                "ready_for_prediction": False,
                "suggestion": "This dataset format is incompatible. Use NSL-KDD format or train a new model for this data structure."
            }
    
    except pd.errors.EmptyDataError:
        return {"valid": False, "error": "CSV file is empty"}
    except pd.errors.ParserError as e:
        return {"valid": False, "error": f"Invalid CSV format: {str(e)}"}
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return {"valid": False, "error": f"Validation failed: {str(e)}"}


@app.get("/dataset-requirements", tags=["Validation"])
async def get_dataset_requirements():
    """
    Get detailed requirements for compatible datasets.
    
    Returns information about expected format, features, and examples.
    """
    return {
        "model_type": "Flexible Network Intrusion Detection",
        "required_format": "CSV",
        "feature_handling": "Adaptive (any feature count supported)",
        "min_features": 3,
        "max_file_size_mb": MAX_FILE_SIZE / (1024*1024),
        "feature_adaptation": {
            "enabled": True,
            "strategies": [
                "Feature padding with statistical features",
                "Feature selection by variance",
                "Automatic feature engineering"
            ]
        },
        "supported_datasets": [
            {
                "name": "nsl_kdd_dataset.csv",
                "features": 41,
                "status": "✅ Optimal (no adaptation needed)"
            },
            {
                "name": "similar_network_dataset.csv", 
                "features": 9,
                "status": "✅ Supported (auto-padding enabled)"
            },
            {
                "name": "log.csv",
                "features": 14,
                "status": "✅ Supported (auto-padding enabled)"
            },
            {
                "name": "Custom datasets",
                "features": "3+",
                "status": "✅ Supported with adaptation"
            }
        ],
        "recommendations": [
            "Any CSV with 3+ numeric features will work",
            "More features = better accuracy",
            "Model automatically adapts to your data",
            "File size should be under 100 MB"
        ],
        "how_it_works": {
            "fewer_features": "System pads with engineered features (mean, std, interactions)",
            "more_features": "System selects most informative features by variance",
            "same_features": "Direct prediction without adaptation"
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


# IP Blocking Management Endpoints
@app.get("/blocked-ips", tags=["Security"])
async def get_blocked_ips():
    """Get list of currently blocked IPs"""
    return {"blocked_ips": ip_blocking_service.get_blocked_ips()}


@app.post("/block-ip", tags=["Security"])
async def block_ip_endpoint(ip: str, reason: str, duration_minutes: int = 60):
    """Manually block an IP address"""
    ip_blocking_service.block_ip(ip, reason, duration_minutes)
    return {"status": "success", "message": f"IP {ip} blocked for {duration_minutes} minutes"}


@app.post("/unblock-ip", tags=["Security"])
async def unblock_ip_endpoint(ip: str):
    """Manually unblock an IP address"""
    success = ip_blocking_service.unblock_ip(ip)
    if success:
        return {"status": "success", "message": f"IP {ip} unblocked"}
    return {"status": "error", "message": f"IP {ip} not found in blocked list"}


# Threat Intel Chatbot Endpoints
@app.post("/chatbot/message", tags=["Chatbot"])
async def chatbot_message(request: dict):
    """Send message to threat intelligence chatbot"""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from chatbot.llm import threat_intel_bot
        
        message = request.get('message', '')
        context = request.get('context', None)
        
        response = threat_intel_bot.chat(message, context)
        return {"response": response, "status": "success"}
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return {"response": "I'm having trouble processing that request. Please try again.", "status": "error"}


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


# Network Testing and Logging Endpoints
@app.post("/network/test", tags=["Network Testing"])
async def run_network_test(
    traffic_type: str = "normal",
    requests_count: int = 50,
    attack_ratio: float = 0.2
):
    """
    Run network traffic test and log results.
    
    Args:
        traffic_type: Type of test (normal, http_flood, port_scan, sql_injection, xss_attack, mixed)
        requests_count: Number of requests to generate
        attack_ratio: Ratio of attack traffic for mixed tests (0.0 to 1.0)
    """
    try:
        tester = NetworkTester(target_url="http://localhost:8000")
        
        if traffic_type == "normal":
            results = await tester.generate_normal_traffic(requests_count)
        elif traffic_type == "http_flood":
            results = await tester.generate_http_flood(requests_count, concurrent=20)
        elif traffic_type == "port_scan":
            results = await tester.generate_port_scan()
        elif traffic_type == "sql_injection":
            results = await tester.generate_sql_injection(requests_count)
        elif traffic_type == "xss_attack":
            results = await tester.generate_xss_attack(requests_count)
        elif traffic_type == "mixed":
            test_results = await tester.run_mixed_test(requests_count, attack_ratio)
            results = test_results["results"]
            
            # Log all results
            for result in results:
                if result.get("traffic_type") == TrafficType.NORMAL.value:
                    network_logger.log_request(
                        source_ip=result.get("source_ip", "unknown"),
                        destination_ip="localhost",
                        source_port=54321,
                        destination_port=8000,
                        protocol="HTTP",
                        request_type=result.get("method", "GET"),
                        response_code=result.get("status"),
                        packet_size=result.get("response_size", 0),
                        duration=result.get("duration", 0)
                    )
                else:
                    network_logger.log_intrusion(
                        source_ip=result.get("source_ip", "unknown"),
                        destination_ip="localhost",
                        intrusion_type=result.get("traffic_type", "Unknown"),
                        confidence=0.85,
                        action="Logged",
                        model_prediction="Attack"
                    )
            
            network_logger.save_to_file()
            
            return {
                "status": "success",
                "message": f"Mixed test completed with {len(results)} requests",
                "summary": test_results["summary"]
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unknown traffic type: {traffic_type}")
        
        # Log results
        for result in results:
            if result.get("traffic_type") == TrafficType.NORMAL.value:
                network_logger.log_request(
                    source_ip=result.get("source_ip", "unknown"),
                    destination_ip="localhost",
                    source_port=54321,
                    destination_port=8000,
                    protocol="HTTP",
                    request_type=result.get("method", "GET"),
                    response_code=result.get("status"),
                    packet_size=result.get("response_size", 0),
                    duration=result.get("duration", 0)
                )
            else:
                network_logger.log_intrusion(
                    source_ip=result.get("source_ip", "unknown"),
                    destination_ip="localhost",
                    intrusion_type=result.get("traffic_type", "Unknown"),
                    confidence=0.85,
                    action="Logged",
                    model_prediction="Attack"
                )
        
        network_logger.save_to_file()
        
        return {
            "status": "success",
            "message": f"{traffic_type} test completed",
            "requests_generated": len(results)
        }
        
    except Exception as e:
        logger.error(f"Network test error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/network/logs", tags=["Network Logging"])
async def get_network_logs(
    limit: int = 100,
    filter_type: Optional[str] = None
):
    """
    Get network traffic logs.
    
    Args:
        limit: Maximum number of logs to return
        filter_type: Filter by type (logged, detected, or None for all)
    """
    try:
        logs = network_logger.get_logs(limit=limit, filter_type=filter_type)
        return {
            "status": "success",
            "count": len(logs),
            "logs": logs
        }
    except Exception as e:
        logger.error(f"Error fetching logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/network/stats", tags=["Network Logging"])
async def get_network_stats():
    """Get network traffic statistics."""
    try:
        stats = network_logger.get_statistics()
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/network/logs", tags=["Network Logging"])
async def clear_network_logs():
    """Clear all network logs."""
    try:
        network_logger.clear_logs()
        return {
            "status": "success",
            "message": "All logs cleared"
        }
    except Exception as e:
        logger.error(f"Error clearing logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# REAL-TIME MONITORING ENDPOINTS
# ============================================================================

@app.post("/realtime/start", tags=["Real-Time Monitoring"])
async def start_realtime_monitoring(use_simulation: bool = True):
    """
    Start real-time network monitoring and ML prediction.
    
    Args:
        use_simulation: If True, uses simulated traffic; if False, waits for real traffic
        
    Returns:
        Status of monitoring service
    """
    try:
        if realtime_service is None:
            raise HTTPException(status_code=500, detail="Real-time service not initialized")
        
        if realtime_service.is_running:
            return {
                "status": "already_running",
                "message": "Real-time monitoring is already active"
            }
        
        await realtime_service.start_monitoring(use_simulation=use_simulation)
        
        logger.info("Real-time monitoring started successfully")
        return {
            "status": "success",
            "message": "Real-time monitoring started",
            "simulation_mode": use_simulation
        }
        
    except Exception as e:
        logger.error(f"Error starting real-time monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/realtime/stop", tags=["Real-Time Monitoring"])
async def stop_realtime_monitoring():
    """
    Stop real-time network monitoring.
    
    Returns:
        Status of monitoring service
    """
    try:
        if realtime_service is None:
            raise HTTPException(status_code=500, detail="Real-time service not initialized")
        
        if not realtime_service.is_running:
            return {
                "status": "not_running",
                "message": "Real-time monitoring is not active"
            }
        
        await realtime_service.stop_monitoring()
        
        logger.info("Real-time monitoring stopped successfully")
        return {
            "status": "success",
            "message": "Real-time monitoring stopped"
        }
        
    except Exception as e:
        logger.error(f"Error stopping real-time monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/status", tags=["Real-Time Monitoring"])
async def get_realtime_status():
    """
    Get current status of real-time monitoring.
    
    Returns:
        Monitoring status and basic metrics
    """
    try:
        if realtime_service is None:
            raise HTTPException(status_code=500, detail="Real-time service not initialized")
        
        metrics = realtime_service.get_metrics()
        
        return {
            "status": "success",
            "is_running": realtime_service.is_running,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting real-time status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/metrics", tags=["Real-Time Monitoring"])
async def get_realtime_metrics():
    """
    Get detailed real-time metrics for dashboard.
    
    Returns:
        Complete metrics including counts, rates, and statistics
    """
    try:
        if realtime_service is None:
            raise HTTPException(status_code=500, detail="Real-time service not initialized")
        
        metrics = realtime_service.get_metrics()
        timeline = realtime_service.get_timeline(num_points=50)
        confidence = realtime_service.get_confidence_distribution()
        
        return {
            "status": "success",
            "metrics": metrics,
            "timeline": timeline,
            "confidence_distribution": confidence
        }
        
    except Exception as e:
        logger.error(f"Error getting real-time metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/visualizations", tags=["Real-Time Monitoring"])
async def get_realtime_visualizations():
    """
    Get all real-time visualization plots.
    
    Returns:
        Dictionary of base64-encoded plot images
    """
    try:
        # Import module to access global variable
        import app.realtime_service as rt_module
        rt_service = rt_module.realtime_service
        
        if rt_service is None:
            # Initialize if not done yet
            initialize_realtime_service(network_logger)
            rt_service = rt_module.realtime_service
            
        if rt_service is None:
            raise HTTPException(status_code=500, detail="Real-time service not initialized")
        
        # Get current metrics
        metrics = rt_service.get_metrics()
        timeline = rt_service.get_timeline(num_points=50)
        confidence = rt_service.get_confidence_distribution()
        
        # Generate visualizations
        plots = realtime_viz_service.generate_realtime_plots_with_timeline(
            metrics, timeline, confidence
        )
        
        return {
            "status": "success",
            "plots": plots,
            "metrics_summary": {
                "total_processed": metrics["total_processed"],
                "total_attacks": metrics["total_attacks"],
                "total_normal": metrics["total_normal"],
                "attack_rate_percent": metrics["attack_rate_percent"],
                "last_update": metrics["last_update"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating real-time visualizations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/realtime/reset", tags=["Real-Time Monitoring"])
async def reset_realtime_metrics():
    """
    Reset all real-time metrics and clear buffers.
    
    Returns:
        Confirmation message
    """
    try:
        if realtime_service is None:
            raise HTTPException(status_code=500, detail="Real-time service not initialized")
        
        realtime_service.reset_metrics()
        
        logger.info("Real-time metrics reset successfully")
        return {
            "status": "success",
            "message": "Real-time metrics reset"
        }
        
    except Exception as e:
        logger.error(f"Error resetting real-time metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/realtime/process-upload", tags=["Real-Time Monitoring"])
async def process_upload_realtime(file: UploadFile = File(...)):
    """
    Process uploaded CSV in real-time mode.
    Simulates real-time processing of historical data.
    
    Args:
        file: CSV file with network traffic data
        
    Returns:
        Processing status
    """
    try:
        # Import module to access global variable
        import app.realtime_service as rt_module
        rt_service = rt_module.realtime_service
        
        if rt_service is None:
            raise HTTPException(status_code=500, detail="Real-time service not initialized")
        
        # Validate file
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
        
        # Read and parse CSV
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        logger.info(f"Processing {len(df)} samples in real-time mode")
        
        # Process in real-time mode (batched)
        await rt_service.process_uploaded_data(df)
        
        # Get updated metrics
        metrics = rt_service.get_metrics()
        
        return {
            "status": "success",
            "message": f"Processed {len(df)} samples in real-time mode",
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error processing upload in real-time: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/attack-types", tags=["Real-Time Monitoring"])
async def get_realtime_attack_types():
    """
    Get detailed breakdown of attack types detected.
    
    Returns:
        Attack type statistics with labels
    """
    try:
        if realtime_service is None:
            raise HTTPException(status_code=500, detail="Real-time service not initialized")
        
        metrics = realtime_service.get_metrics()
        attack_counts = metrics["attack_type_counts_all"]
        
        # Format with labels
        attack_types = []
        for attack_id, count in attack_counts.items():
            attack_types.append({
                "id": attack_id,
                "type": MULTICLASS_LABELS.get(attack_id, "Unknown"),
                "count": count,
                "percentage": round(count / metrics["total_processed"] * 100, 2) 
                            if metrics["total_processed"] > 0 else 0
            })
        
        return {
            "status": "success",
            "attack_types": attack_types,
            "total_processed": metrics["total_processed"]
        }
        
    except Exception as e:
        logger.error(f"Error getting attack types: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/realtime/ws")
async def realtime_websocket(websocket):
    """
    WebSocket endpoint for real-time dashboard updates.
    Pushes metrics every second while connected.
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            if realtime_service is None:
                await websocket.send_json({
                    "error": "Real-time service not initialized"
                })
                break
            
            # Get current metrics
            metrics = realtime_service.get_metrics()
            
            # Send to client
            await websocket.send_json({
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics
            })
            
            # Wait 1 second before next update
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info("WebSocket connection closed")
