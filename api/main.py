"""Containerized Audience Segmentation & Personalization Service - Phase 2 API.

A lightweight, production-ready REST API built with FastAPI. It loads the persisted
Phase 1 machine learning pipeline (StandardScaler + KMeans) and cluster metadata at
startup to serve real-time audience segment predictions, distance-to-centroid metrics,
and personalized content recommendations without retraining.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ott_api")

# ---------------------------------------------------------------------------
# File Path Resolution
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "pipeline.pkl")))
METADATA_PATH = Path(os.getenv("METADATA_PATH", str(BASE_DIR / "models" / "metadata.json")))

# ---------------------------------------------------------------------------
# Static Fictional Recommendation Catalog
# ---------------------------------------------------------------------------
CATALOG: Dict[str, List[str]] = {
    "High-Engagement Power Viewers": [
        "Action Horizon",
        "Midnight Chase",
        "Final Mission",
        "Apex Predator",
    ],
    "Weekend Binge Viewers": [
        "Weekend Escape",
        "The Silent Harbor",
        "City Stories",
        "Deep Investigation",
    ],
    "Casual Viewers": [
        "Comedy Nights",
        "Laugh Out Loud",
        "Quick Laughs",
        "Bite-Sized Chronicles",
    ],
    "Genre-Focused Viewers (Action & Sci-Fi)": [
        "Galaxy Frontier",
        "Cyber Strike",
        "Nebula Rising",
        "Hyperdrive Zero",
    ],
    "Low-Activity / Churn-Risk Viewers": [
        "Trending Now: Top 10",
        "Quick Catchup Series",
        "The Return",
        "Stream Highlights",
    ],
}

DEFAULT_RECOMMENDATIONS = [
    "Trending Now: Top 10",
    "Weekend Escape",
    "Comedy Nights",
]


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service status ('ok' or 'not_ready')")
    model_loaded: bool = Field(..., description="Whether the ML model pipeline is loaded")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "model_loaded": True,
            }
        }
    )


class ViewerProfileRequest(BaseModel):
    """Viewer behavioral profile input schema for audience segmentation.

    Field constraints match Phase 1 data validation rules.
    """

    user_id: str = Field(
        ...,
        min_length=1,
        description="Unique non-empty viewer identifier (excluded from ML features)",
        examples=["USR-8192"],
    )
    watch_time_hours: float = Field(
        ...,
        ge=0.0,
        description="Total weekly viewing time in hours (must be >= 0)",
        examples=[32.5],
    )
    sessions_per_week: float = Field(
        ...,
        ge=0.0,
        description="Number of viewing sessions per week (must be >= 0)",
        examples=[5.0],
    )
    avg_session_mins: float = Field(
        ...,
        gt=0.0,
        description="Average length of viewing sessions in minutes (must be > 0)",
        examples=[85.0],
    )
    completion_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Content completion ratio between 0.0 and 1.0",
        examples=[0.82],
    )
    action_pct: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of viewing devoted to Action genre [0.0, 1.0]",
        examples=[0.45],
    )
    comedy_pct: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of viewing devoted to Comedy genre [0.0, 1.0]",
        examples=[0.10],
    )
    drama_pct: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of viewing devoted to Drama genre [0.0, 1.0]",
        examples=[0.15],
    )
    thriller_pct: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of viewing devoted to Thriller genre [0.0, 1.0]",
        examples=[0.20],
    )
    sci_fi_pct: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of viewing devoted to Sci-Fi genre [0.0, 1.0]",
        examples=[0.10],
    )
    weekend_viewing_pct: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of total viewing during weekends [0.0, 1.0]",
        examples=[0.65],
    )
    days_since_last_watch: float = Field(
        ...,
        ge=0.0,
        description="Inactivity duration in days since last playback (must be >= 0)",
        examples=[2.0],
    )

    @field_validator("user_id")
    @classmethod
    def validate_non_whitespace_user_id(cls, v: str) -> str:
        """Ensure user_id is not just whitespace."""
        if not v.strip():
            raise ValueError("user_id cannot be an empty or whitespace string")
        return v.strip()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "USR-8192",
                "watch_time_hours": 32.5,
                "sessions_per_week": 5.0,
                "avg_session_mins": 85.0,
                "completion_rate": 0.82,
                "action_pct": 0.45,
                "comedy_pct": 0.10,
                "drama_pct": 0.15,
                "thriller_pct": 0.20,
                "sci_fi_pct": 0.10,
                "weekend_viewing_pct": 0.65,
                "days_since_last_watch": 2.0,
            }
        }
    )


class RecommendationResponse(BaseModel):
    """Personalized recommendation response schema."""

    user_id: str = Field(..., description="Viewer identifier")
    segment_id: int = Field(..., description="Assigned K-Means cluster identifier (0 to K-1)")
    segment_name: str = Field(..., description="Human-readable behavioral audience segment name")
    recommendations: List[str] = Field(
        ...,
        description="Curated list of personalized title recommendations based on segment profile",
    )
    distance_to_centroid: float = Field(
        ...,
        description=(
            "Euclidean distance between the user's standardized feature vector and the centroid "
            "of the assigned cluster in scaled feature space. Smaller distance indicates closer "
            "alignment with the prototypical behavior of that segment (not a probability)."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "USR-8192",
                "segment_id": 4,
                "segment_name": "Genre-Focused Viewers (Action & Sci-Fi)",
                "recommendations": [
                    "Galaxy Frontier",
                    "Cyber Strike",
                    "Nebula Rising",
                ],
                "distance_to_centroid": 0.42,
            }
        }
    )


# ---------------------------------------------------------------------------
# Lifespan Management: Single Startup Model Load
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager.

    Loads the persisted scikit-learn pipeline and metadata artifact once during
    application startup into application state. Retraining is strictly prohibited.
    """
    logger.info("Initializing OTT Audience Intelligence API service...")
    app.state.model_loaded = False
    app.state.model_artifact = None
    app.state.pipeline = None
    app.state.scaler = None
    app.state.kmeans = None
    app.state.metadata = None
    app.state.feature_names = None

    try:
        if not MODEL_PATH.exists():
            logger.error("Model artifact not found at: %s", MODEL_PATH)
        elif not METADATA_PATH.exists():
            logger.error("Metadata artifact not found at: %s", METADATA_PATH)
        else:
            # 1. Load joblib model artifact
            artifact = joblib.load(MODEL_PATH)
            logger.info("Loaded pipeline artifact from: %s", MODEL_PATH)

            # 2. Load JSON metadata
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            logger.info("Loaded metadata artifact from: %s", METADATA_PATH)

            # 3. Extract and validate components
            pipeline = artifact["pipeline"]
            scaler = pipeline.named_steps["scaler"]
            kmeans = pipeline.named_steps["kmeans"]
            feature_names = artifact["feature_names"]

            # Store in application state
            app.state.model_artifact = artifact
            app.state.pipeline = pipeline
            app.state.scaler = scaler
            app.state.kmeans = kmeans
            app.state.metadata = metadata
            app.state.feature_names = feature_names
            app.state.model_loaded = True

            logger.info(
                "Model successfully loaded. Selected K=%d, Feature Count=%d",
                kmeans.n_clusters,
                len(feature_names),
            )
    except Exception as exc:
        logger.exception("Failed to load Phase 1 model artifact during startup: %s", exc)
        app.state.model_loaded = False

    yield

    logger.info("Shutting down OTT Audience Intelligence API service.")


# ---------------------------------------------------------------------------
# FastAPI Application Declaration
# ---------------------------------------------------------------------------
app = FastAPI(
    title="OTT Audience Segmentation & Personalization API",
    description=(
        "Production-style REST API serving unsupervised machine learning audience "
        "segmentation and personalized recommendations for an OTT streaming platform. "
        "Built on top of persisted Phase 1 StandardScaler + KMeans artifacts."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configurable CORS (production-safe default supporting frontend containers & local dev)
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
if allowed_origins_env:
    extra_origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]
    origins = list(set(default_origins + extra_origins))
else:
    origins = default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Custom Error Handlers (Clean JSON, No Stack Traces Exposed)
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Format Pydantic validation errors cleanly without internal stack traces."""
    errors = []
    for err in exc.errors():
        field_path = " -> ".join(str(loc) for loc in err.get("loc", []))
        errors.append(
            {
                "field": field_path,
                "message": err.get("msg"),
                "type": err.get("type"),
            }
        )
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Input viewer profile failed validation constraints.",
            "details": errors,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught errors safely without leaking internal paths or traces."""
    logger.exception("Unexpected server error processing request: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred while processing the request.",
        },
    )


# ---------------------------------------------------------------------------
# Helper: Feature Preparation & Personalization Rules
# ---------------------------------------------------------------------------
def prepare_features(
    request_data: ViewerProfileRequest, feature_names: List[str]
) -> pd.DataFrame:
    """Construct a single-row DataFrame using the EXACT feature names and order

    established during Phase 1 training.
    """
    feature_dict = {
        "watch_time_hours": [request_data.watch_time_hours],
        "sessions_per_week": [request_data.sessions_per_week],
        "avg_session_mins": [request_data.avg_session_mins],
        "completion_rate": [request_data.completion_rate],
        "action_pct": [request_data.action_pct],
        "comedy_pct": [request_data.comedy_pct],
        "drama_pct": [request_data.drama_pct],
        "thriller_pct": [request_data.thriller_pct],
        "sci_fi_pct": [request_data.sci_fi_pct],
        "weekend_viewing_pct": [request_data.weekend_viewing_pct],
        "days_since_last_watch": [request_data.days_since_last_watch],
    }

    # Guarantees column ordering strictly matches Phase 1 training
    df = pd.DataFrame(feature_dict)[feature_names]
    return df


def get_personalization_recommendations(
    segment_name: str, profile: ViewerProfileRequest
) -> List[str]:
    """Transparent rule-based personalization layer.

    Generates curated title recommendations based on the predicted segment, with
    genre-aware fine-tuning where appropriate.
    """
    base_titles = CATALOG.get(segment_name, DEFAULT_RECOMMENDATIONS)
    recommendations = list(base_titles[:3])

    # If the user has a dominant genre that isn't already covered, optionally highlight a match
    genre_scores = {
        "Action": profile.action_pct,
        "Comedy": profile.comedy_pct,
        "Drama": profile.drama_pct,
        "Thriller": profile.thriller_pct,
        "Sci-Fi": profile.sci_fi_pct,
    }
    dominant_genre, top_score = max(genre_scores.items(), key=lambda x: x[1])

    if top_score >= 0.40 and dominant_genre == "Comedy" and "Comedy Nights" not in recommendations:
        recommendations[2] = "Comedy Nights"
    elif top_score >= 0.40 and dominant_genre == "Sci-Fi" and "Galaxy Frontier" not in recommendations:
        recommendations[2] = "Galaxy Frontier"

    return recommendations


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/",
    summary="Root Service Index",
    tags=["General"],
)
async def root() -> Dict[str, Any]:
    """Root endpoint providing quick navigation to documentation and health."""
    return {
        "service": "OTT Audience Segmentation & Personalization Service",
        "status": "online" if getattr(app.state, "model_loaded", False) else "model_unavailable",
        "swagger_docs": "/docs",
        "health_check": "/health",
        "recommendation_endpoint": "/recommend",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health & Model Readiness Check",
    tags=["Health"],
    responses={
        200: {"description": "Service is healthy and ML model is ready for inference."},
        503: {"description": "Service is unready because the ML model is not loaded."},
    },
)
async def health_check(response: Response) -> Dict[str, Any]:
    """Check the health and readiness of the segmentation service.

    Returns:
    - HTTP 200 with `model_loaded: true` if the persisted pipeline is ready.
    - HTTP 503 with `model_loaded: false` if the model failed to load.
    """
    if not app.state.model_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready", "model_loaded": False}

    return {"status": "ok", "model_loaded": True}


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Audience Segmentation & Personalized Recommendations",
    tags=["Segmentation & Personalization"],
    responses={
        200: {"description": "Segment predicted and recommendations generated successfully."},
        422: {"description": "Input validation failure (e.g. negative numbers, out-of-range ratios)."},
        503: {"description": "Model is not loaded or unavailable."},
    },
)
async def recommend(request_data: ViewerProfileRequest) -> Dict[str, Any]:
    """Segment a viewer profile using the persisted Phase 1 K-Means pipeline

    and generate personalized content recommendations.

    Execution Flow:
    1. Validate input fields using Pydantic.
    2. Check model readiness (HTTP 503 if unready; no retraining).
    3. Construct input vector in exact Phase 1 training feature order.
    4. Predict cluster ID using the persisted scikit-learn pipeline.
    5. Calculate Euclidean distance from user's scaled profile to the cluster centroid.
    6. Translate cluster ID to human-readable segment name from metadata.
    7. Generate transparent rule-based recommendations.
    """
    # 1. Model Readiness Guard
    if not app.state.model_loaded:
        logger.error("POST /recommend requested while model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Clustering model artifact is not loaded. Service is not ready.",
        )

    try:
        # 2. Prepare Features in Exact Phase 1 Order
        feature_names: List[str] = app.state.feature_names
        df_features = prepare_features(request_data, feature_names)

        # 3. K-Means Inference (Read-Only Prediction, Retraining is Strictly Prohibited)
        pipeline = app.state.pipeline
        scaler = app.state.scaler
        kmeans = app.state.kmeans

        cluster_id = int(pipeline.predict(df_features)[0])

        # 4. Compute Distance to Centroid in Scaled Feature Space
        x_scaled = scaler.transform(df_features)  # shape (1, 11)
        centroid = kmeans.cluster_centers_[cluster_id]  # shape (11,)
        euclidean_dist = float(np.linalg.norm(x_scaled[0] - centroid))
        distance_to_centroid = round(euclidean_dist, 4)

        # 5. Resolve Human-Readable Segment Name from Metadata
        segment_names: Dict[int, str] = app.state.model_artifact.get(
            "segment_names", {}
        )
        segment_name = segment_names.get(cluster_id, f"Cluster {cluster_id}")

        # 6. Apply Personalization Rules
        recommendations = get_personalization_recommendations(
            segment_name, request_data
        )

        logger.info(
            "Inference completed for user_id=%s -> segment_id=%d (%s), distance=%.4f",
            request_data.user_id,
            cluster_id,
            segment_name,
            distance_to_centroid,
        )

        return {
            "user_id": request_data.user_id,
            "segment_id": cluster_id,
            "segment_name": segment_name,
            "recommendations": recommendations,
            "distance_to_centroid": distance_to_centroid,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Internal error during segmentation prediction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during cluster prediction.",
        )


@app.get(
    "/metadata",
    summary="Model & Cluster Metadata",
    tags=["Metadata"],
    responses={
        200: {"description": "Phase 1 model training metadata and cluster profiles."},
        503: {"description": "Model is not loaded or unavailable."},
    },
)
async def get_metadata() -> Dict[str, Any]:
    """Retrieve Phase 1 model metadata, cluster centroid profiles, and recommendation catalog.

    Safe read-only endpoint powering frontend analytics and explainability.
    """
    if not app.state.model_loaded or not app.state.metadata:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is not loaded.",
        )
    return {
        "metadata": app.state.metadata,
        "catalog": CATALOG,
        "feature_names": app.state.feature_names,
    }
