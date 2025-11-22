"""
Pydantic models for request/response validation.
Defines the structure of data sent to and from the API.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class PredictionResponse(BaseModel):
    """Response model for a single prediction."""
    sample_id: int
    binary_prediction: str
    binary_confidence: float
    multiclass_prediction: str
    multiclass_confidence: float


class MetricsResponse(BaseModel):
    """Response model for model metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]


class ClassDistribution(BaseModel):
    """Distribution of predicted classes."""
    label: str
    count: int
    percentage: float


class PredictionSummary(BaseModel):
    """Summary of all predictions."""
    total_samples: int
    binary_distribution: List[ClassDistribution]
    multiclass_distribution: List[ClassDistribution]
    attack_percentage: float
    normal_percentage: float


class ReportResponse(BaseModel):
    """Complete response with predictions and analysis."""
    summary: PredictionSummary
    predictions: List[PredictionResponse]
    binary_metrics: Optional[MetricsResponse] = None
    multiclass_metrics: Optional[MetricsResponse] = None
    plots: Dict[str, str]  # plot_name -> base64_encoded_image
    message: str = "Analysis complete"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str
    models_loaded: Dict[str, bool]
    version: str
