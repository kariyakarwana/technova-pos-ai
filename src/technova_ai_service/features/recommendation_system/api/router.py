"""FastAPI router for TechNova AI Product Recommendation Engine."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from ..application.service import create_recommendations
from .schemas import (
    RecommendationRequest,
    RecommendationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/recommendations", tags=["recommendations"])


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate ranked, stock-aware product recommendations across retail contexts",
)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    """Generate ranked product recommendations for TechNova POS.

    Supported contexts:
    - **CUSTOMER**: Personalized recommendations tailored to loyalty purchase history and brand affinity.
    - **PRODUCT**: Up-sell, cross-sell, and complementary accessory discovery for a selected product.
    - **BRANCH**: Localized assortment ranking with empirical Bayesian smoothing.
    - **TRENDING**: High-velocity momentum products accelerating in sales.
    - **COLD_START**: Robust fallback recommendations for anonymous walk-ins or new branches.
    - **CART_READY**: Real-time basket completion recommendations for multiple items in cart.

    Returns:
    - Ranked products with normalized confidence scores
    - Machine-readable `reason_code` and human-readable `reason`
    - Physical branch stock availability
    - Model provenance metadata
    """
    try:
        return create_recommendations(request)
    except FileNotFoundError as error:
        logger.error("Recommendation artifact missing: %s", error)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Recommendation service unavailable: {error}",
        ) from error
    except ValueError as error:
        logger.warning("Invalid recommendation request: %s", error)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except Exception:
        logger.exception("Unexpected error in recommendation router")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Recommendation inference failed unexpectedly.",
        ) from None
