def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_predict_ok(client):
    res = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert res.status_code == 200
    body = res.json()
    assert body["label"] == "setosa"
    assert 0 <= body["confidence"] <= 1


def test_predict_wrong_length_returns_422(client):
    res = client.post("/predict", json={"features": [1, 2]})
    assert res.status_code == 422


def test_predict_missing_field_returns_422(client):
    res = client.post("/predict", json={})
    assert res.status_code == 422
