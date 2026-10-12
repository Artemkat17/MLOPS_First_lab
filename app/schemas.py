from typing import Literal

from pydantic import BaseModel, Field, field_validator

SentimentLabel = Literal["positive", "neutral", "negative"]


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        description="English text to analyze",
        examples=["I love this product!"],
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must not be blank")
        return value


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        description="List of texts to analyze",
        examples=[["Great product!", "Terrible service."]],
    )

    @field_validator("texts")
    @classmethod
    def texts_must_not_contain_blank_values(
        cls, values: list[str]
    ) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("Texts must not contain blank values")
        return values


class Prediction(BaseModel):
    label: SentimentLabel
    score: float = Field(ge=0.0, le=1.0)


class BatchPredictResponse(BaseModel):
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    task: str
    labels: list[str]
    transformers_version: str
    torch_version: str