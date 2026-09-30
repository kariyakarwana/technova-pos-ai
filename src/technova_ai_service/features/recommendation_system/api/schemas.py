"""Pydantic schemas for the TechNova AI Product Recommendation Engine."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..domain.enums import RecommendationContext, RecommendationReasonCode


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organization_id: str = Field(
        ...,
        min_length=1,
        description="Tenant organization identifier for strict multi-tenant isolation.",
        examples=["org_technova_default"],
    )
    context: RecommendationContext = Field(
        ...,
        description="The operational POS context triggering the recommendation request.",
        examples=["PRODUCT"],
    )
    branch_id: str | None = Field(
        default=None,
        description="The physical retail branch identifier for inventory stock-aware filtering.",
        examples=["BRANCH-001"],
    )
    customer_id: str | None = Field(
        default=None,
        description="Synthetic customer identifier for personalized recommendations.",
        examples=["CUST-0001"],
    )
    product_id: str | None = Field(
        default=None,
        description="Target product SKU/ID when context is PRODUCT.",
        examples=["PROD-COMP-001"],
    )
    product_ids: list[str] | None = Field(
        default=None,
        description="List of target product SKUs/IDs when context is CART_READY.",
        examples=[["PROD-COMP-001", "PROD-PERI-001"]],
    )
    top_n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of ranked recommendations to return.",
    )
    include_substitutes: bool = Field(
        default=False,
        description="Whether to return substitute items if exact matches are out of stock.",
    )
    category: str | None = Field(
        default=None,
        description="Category of the target product from runtime DB.",
    )
    product_name: str | None = Field(
        default=None,
        description="Name of the target product from runtime DB.",
    )
    brand: str | None = Field(
        default=None,
        description="Brand of the target product from runtime DB.",
    )
    price: float | None = Field(
        default=None,
        description="Price of the target product from runtime DB.",
    )

    @model_validator(mode="after")
    def validate_context_requirements(self) -> RecommendationRequest:
        """Validates context-specific field requirements and rejects incomplete requests."""
        ctx = self.context

        if ctx == RecommendationContext.PRODUCT:
            if not self.product_id or not self.product_id.strip():
                raise ValueError("product_id is required when context is PRODUCT.")

        elif ctx == RecommendationContext.CART_READY:
            if not self.product_ids or len(self.product_ids) == 0:
                raise ValueError("product_ids list with at least 1 item is required when context is CART_READY.")

        elif ctx == RecommendationContext.BRANCH:
            if not self.branch_id or not self.branch_id.strip():
                raise ValueError("branch_id is required when context is BRANCH.")

        elif ctx == RecommendationContext.CUSTOMER and (
            not self.customer_id or not self.customer_id.strip()
        ):
            raise ValueError("customer_id is required when context is CUSTOMER.")

        return self


class RecommendedProduct(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(..., description="Unique product SKU/identifier.")
    name: str | None = Field(default=None, description="Human-readable product name.")
    category: str | None = Field(default=None, description="Top-level or primary category.")
    brand: str | None = Field(default=None, description="Product brand name.")
    price: float | None = Field(default=None, description="Current selling unit price.")
    score: float = Field(..., description="Normalized recommendation confidence score in [0.0, 1.0].")
    reason_code: RecommendationReasonCode = Field(
        ...,
        description="Machine-readable deterministic reason token.",
    )
    reason: str = Field(..., description="Human-readable cashier/customer explanation.")
    stock_quantity: int = Field(..., description="Physical available stock units at the target branch.")
    is_available: bool = Field(..., description="Boolean indicating whether item is in-stock and active.")


class RecommendationModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(..., description="Recommendation model name.")
    version: str = Field(..., description="Semantic version of recommendation artifact.")
    algorithms_used: list[str] = Field(..., description="List of algorithms contributing to recommendation.")
    context: str = Field(..., description="Evaluated recommendation context.")
    total_candidates_scored: int = Field(..., description="Number of candidate items evaluated before filtering.")
    training_date_range: dict[str, str] = Field(default_factory=dict, description="Date bounds of training data.")


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context: RecommendationContext = Field(..., description="Recommendation context processed.")
    organization_id: str = Field(..., description="Tenant organization identifier.")
    branch_id: str | None = Field(default=None, description="Target branch identifier if scoped.")
    recommendations: list[RecommendedProduct] = Field(
        ...,
        description="Ranked list of product recommendations ordered by score descending.",
    )
    generated_at: str = Field(..., description="ISO 8601 UTC timestamp of inference execution.")
    model_metadata: RecommendationModelMetadata = Field(
        ...,
        description="Model configuration and provenance metadata.",
    )


__all__ = [
    "RecommendationContext",
    "RecommendationModelMetadata",
    "RecommendationReasonCode",
    "RecommendationRequest",
    "RecommendationResponse",
    "RecommendedProduct",
]
