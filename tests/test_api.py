from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    """Проверяет доступность API."""
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"


def test_predict():
    """Проверяет реальное предсказание модели для одного текста."""
    response = client.post(
        "/predict",
        json={"text": "I absolutely love this product! It is amazing."},
    )

    assert response.status_code == 200

    data = response.json()

    assert "label" in data
    assert "score" in data
    assert data["label"] in {"positive", "neutral", "negative"}
    assert isinstance(data["score"], (int, float))
    assert 0 <= data["score"] <= 1


def test_predict_batch():
    """Проверяет реальное предсказание для нескольких текстов."""
    texts = [
        "I absolutely love this product!",
        "This product is terrible. I hate it.",
    ]

    response = client.post(
        "/predict_batch",
        json={"texts": texts},
    )

    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == len(texts)

    for prediction in data["predictions"]:
        assert "label" in prediction
        assert "score" in prediction
        assert prediction["label"] in {
            "positive",
            "neutral",
            "negative",
        }
        assert isinstance(prediction["score"], (int, float))
        assert 0 <= prediction["score"] <= 1


def test_model_info():
    """Проверяет получение информации о модели."""
    response = client.get("/model_info")

    assert response.status_code == 200

    data = response.json()
    assert "model_name" in data


def test_predict_empty_text():
    """Проверяет обработку пустого текста."""
    response = client.post(
        "/predict",
        json={"text": "   "},
    )

    # Ожидается 422, если схема API запрещает пустой текст.
    assert response.status_code == 422
