from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI

from app.model import MODEL_NAME, predict_text
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    Prediction,
    PredictRequest,
)

app = FastAPI(
    title="Sentiment Analysis API",
    description="REST API for text sentiment classification",
    version="1.0.0",
)


def get_package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=Prediction)
def predict(request: PredictRequest) -> Prediction:
    result = predict_text(request.text)
    return Prediction(**result)


@app.post(
    "/predict_batch",
    response_model=BatchPredictResponse,
)
def predict_batch(
    request: BatchPredictRequest,
) -> BatchPredictResponse:
    predictions = [Prediction(**predict_text(text)) for text in request.texts]
    return BatchPredictResponse(predictions=predictions)


@app.get("/model_info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        task="sentiment-analysis",
        labels=["negative", "neutral", "positive"],
        transformers_version=get_package_version("transformers"),
        torch_version=get_package_version("torch"),
    )
