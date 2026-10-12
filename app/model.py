from functools import lru_cache

from transformers import pipeline

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


@lru_cache(maxsize=1)
def get_pipeline():
    """Load the model once and reuse it."""
    return pipeline(
        task="text-classification",
        model=MODEL_NAME,
        device=-1,
    )


def predict_text(text: str) -> dict[str, str | float]:
    """Predict the sentiment of a single text."""
    result = get_pipeline()(text, truncation=True)[0]

    return {
        "label": result["label"].lower(),
        "score": float(result["score"]),
    }