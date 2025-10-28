from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "status": "ok",
        "message": "Alphabet Sign Recognition API is running",
    }
