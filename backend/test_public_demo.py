from app import fastapi_app, get_cvat_config
from fastapi.testclient import TestClient


def test_public_demo_blocks_shared_credential_writes(monkeypatch):
    monkeypatch.setenv("QC_PUBLIC_DEMO", "true")
    client = TestClient(fastapi_app)

    gemini = client.post("/api/gemini/config", json={"api_key": "test-key"})
    cvat = client.post("/api/cvat/config", json={"cvat_url": "https://example.invalid", "cvat_access_token": "test"})

    assert gemini.status_code == 403
    assert cvat.status_code == 403
    assert get_cvat_config() == ("", "")


def test_local_mode_keeps_primary_ui_and_health_route_available(monkeypatch):
    monkeypatch.setenv("QC_PUBLIC_DEMO", "false")
    client = TestClient(fastapi_app)

    assert client.get("/api/health").status_code == 200
    assert client.get("/").status_code == 200
