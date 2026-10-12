from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_result(monkeypatch):
    def fake_predict(text):
        return {
            "label": "positive",
            "score": 0.95,
        }

    monkeypatch.setattr("app.main.predict_text", fake_predict)

    response = client.post(
        "/predict",
        json={"text": "I love this product!"},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["label"] == "positive"
    assert data["score"] == 0.95


def test_predict_batch_processes_multiple_texts(monkeypatch):
    def fake_predict(text):
        label = "positive" if "love" in text else "negative"
        return {
            "label": label,
            "score": 0.9,
        }

    monkeypatch.setattr("app.main.predict_text", fake_predict)

    response = client.post(
        "/predict_batch",
        json={
            "texts": [
                "I love this movie!",
                "I hate this movie!",
            ]
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["label"] == "positive"
    assert data["predictions"][1]["label"] == "negative"


def test_predict_batch_accepts_empty_list():
    response = client.post(
        "/predict_batch",
        json={"texts": []},
    )

    assert response.status_code == 200
    assert response.json() == {"predictions": []}


def test_predict_rejects_blank_text():
    response = client.post(
        "/predict",
        json={"text": "   "},
    )

    assert response.status_code == 422
