"""Regression coverage for stalls, provider routing, and wrong-image comparisons."""
import asyncio
import base64
import threading
import time
import urllib.error

import httpx
import pytest

import app
import qc_pipeline
from ai_engine import local_model_providers as providers
from ai_engine.guideline_manager import GuidelineManager


@pytest.mark.parametrize("path", ["/api/models", "/api/qc/run"])
def test_health_stays_responsive_during_slow_model_work(monkeypatch, path):
    started = threading.Event()
    release = threading.Event()

    def slow_work(*args, **kwargs):
        started.set()
        release.wait(3)
        return [] if path == "/api/models" else {"sorted_frames": []}

    monkeypatch.setattr(app, "available_providers", slow_work)
    monkeypatch.setattr(app, "run_bbox_pipeline", slow_work)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.fastapi_app), base_url="http://test") as client:
            begin = time.monotonic()
            request = client.get(path) if path == "/api/models" else client.post(path, json={"task_type": "bbox", "data": []})
            pending = asyncio.create_task(request)
            try:
                assert await asyncio.to_thread(started.wait, 1)
                response = await asyncio.wait_for(client.get("/api/health"), timeout=1)
                assert response.status_code == 200
                assert time.monotonic() - begin < 2, "Model work blocked the API event loop"
            finally:
                release.set()
                await pending

    asyncio.run(scenario())


def test_upload_wins_over_same_named_demo_image(monkeypatch, tmp_path):
    demo = tmp_path / "same.jpg"
    demo.write_bytes(b"wrong demo image")
    monkeypatch.setattr(app, "find_image_file_path", lambda _: str(demo))
    uploaded = base64.b64encode(b"actual uploaded image").decode()
    assert qc_pipeline._image_bytes("same.jpg", {"same.jpg": uploaded}) == b"actual uploaded image"


def test_dimensions_follow_frame_after_priority_sort(monkeypatch):
    from qc_engine import BDDQualityChecker
    monkeypatch.setattr(BDDQualityChecker, "check_dataset_advanced", lambda *_: {
        "sorted_frames": [{"frame_name": "b.jpg", "frame_index": 1}, {"frame_name": "a.jpg", "frame_index": 0}]})
    result = qc_pipeline._base_bbox_frames(
        [{"name": "a.jpg", "width": 640, "height": 480}, {"name": "b.jpg", "width": 1280, "height": 720}],
        {"annotation_format": "bdd100k"}, GuidelineManager(""))
    assert [(frame["width"], frame["height"]) for frame in result] == [(1280, 720), (640, 480)]


def test_modal_does_not_enable_unrelated_remote_endpoints(monkeypatch):
    monkeypatch.setenv("QC_MODAL_API_URL", "https://example.modal.run")
    monkeypatch.delenv("QC_ALLOW_REMOTE_MODEL_ENDPOINTS", raising=False)
    assert providers._url("bbox") == "https://example.modal.run/bbox"
    assert "QC_ALLOW_REMOTE_MODEL_ENDPOINTS" not in providers.os.environ
    with pytest.raises(providers.ProviderError):
        providers._url("dinov3_seg")


def test_modal_failure_is_not_retried_as_other_models(monkeypatch):
    monkeypatch.setenv("QC_MODAL_API_URL", "https://example.modal.run")
    calls = []

    def fail(capability, *args, **kwargs):
        calls.append(capability)
        raise providers.ProviderError("unavailable")

    monkeypatch.setattr(providers, "_post", fail)
    with pytest.raises(providers.ProviderError):
        providers.infer_bbox(b"image", 0.25, ["car"])
    assert calls == ["bbox"]


def test_offline_modal_is_not_reported_available(monkeypatch):
    monkeypatch.setenv("QC_MODAL_API_URL", "https://offline.modal.run")
    monkeypatch.setattr(providers, "_STATUS_CACHE", {})

    def fail(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(providers.urllib.request, "urlopen", fail)
    result = providers.available_providers()
    assert [item["id"] for item in result] == ["bbox", "segmentation"]
    assert all(not item["available"] for item in result)


def test_public_gemini_rejected_before_inference(monkeypatch):
    monkeypatch.setenv("QC_PUBLIC_DEMO", "true")
    monkeypatch.setattr(app, "run_bbox_pipeline", lambda *a, **k: pytest.fail("should reject before model work"))
    with pytest.raises(app.HTTPException) as exc:
        app.run_workspace_qc_payload({"task_type": "bbox", "use_gemini": True})
    assert exc.value.status_code == 403
