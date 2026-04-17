def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200


def test_predict_validation(client):
    response = client.post("/api/predict", json={"power": -1, "time": 2, "antenna_type": "Other"})
    assert response.status_code in (400, 422)
