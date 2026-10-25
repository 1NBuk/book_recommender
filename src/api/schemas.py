from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Book(BaseModel):
    item_id: str
    isbn: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    year: Optional[int] = None
    publisher: Optional[str] = None
    image_url: Optional[str] = None
    average_rating: Optional[float] = None


class Recommendation(Book):
    rank: int = Field(..., ge=1)
    score: float


class RecommendationResponse(BaseModel):
    user_id: str
    # personalized      - the user has precomputed recommendations from the model
    # popular_fallback  - cold-start user, we return globally popular books
    source: Literal["personalized", "popular_fallback"]
    model: str
    count: int
    items: list[Recommendation]


class PopularResponse(BaseModel):
    count: int
    items: list[Recommendation]


class ModelInfo(BaseModel):
    model: str
    created_at: Optional[str] = None
    n_users: int
    n_items: int
    n_popular: int
    metrics: dict[str, float] = {}


class HealthResponse(BaseModel):
    status: str
