import json
import urllib.request
import io
import pytest
from unittest.mock import MagicMock
from app import push_issues_to_cvat_payload, fastapi_app


def test_push_issues_to_cvat_payload_success(monkeypatch):
    calls = []

    def mock_urlopen(req, timeout=None):
        url = req.full_url
        method = req.get_method()
        calls.append((method, url))
        
        resp = MagicMock()
        resp.status = 200
        if "/api/jobs/1" in url and "/data/meta" not in url:
            data = {"id": 1, "task_id": 10, "stage": "annotation", "state": "in progress"}
        elif "/api/jobs/1/data/meta" in url:
            data = {"frames": [{"frame": 0, "name": "frame_0.jpg"}]}
        elif "/api/issues?job_id=1" in url:
            data = {"results": []}
        elif "/api/issues" in url and method == "POST":
            resp.status = 201
            data = {"id": 999, "frame": 0, "comments": [{"message": "created"}]}
        else:
            data = {}

        resp.read.return_value = json.dumps(data).encode("utf-8")
        resp.__enter__.return_value = resp
        return resp

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    payload = {
        "cvat_url": "https://app.cvat.ai",
        "cvat_access_token": "valid_token_123",
        "job_id": 1,
        "frame_offset": 0,
        "frame_map": {"frame_0.jpg": 0},
        "confirm_offset_mapping": True,
        "include_all_unreviewed": True,
        "update_job_state": False,
        "sorted_frames": [
            {
                "frame_name": "frame_0.jpg",
                "frame_index": 0,
                "labels": [{"id": "obj_1", "box2d": {"x1": 10, "y1": 20, "x2": 50, "y2": 60}}],
                "errors": [
                    {
                        "issue_id": "test_issue_1",
                        "error_type": "MISSING_OBJECT",
                        "severity": "HIGH",
                        "explanation": "Annotator missed vehicle",
                        "object_id": "obj_1",
                        "suggested_box": [10.0, 20.0, 50.0, 60.0],
                        "review_status": "confirmed",
                    }
                ]
            }
        ]
    }

    req = MagicMock()
    req.headers = {}
    result = push_issues_to_cvat_payload(req, payload)

    assert result["status"] == "success"
    assert result["created"] == 1
    assert result["failed"] == []
    assert any(method == "POST" and "/api/issues" in url for method, url in calls)


def test_return_job_zero_issues_and_update_state(monkeypatch):
    calls = []
    current_job = {"id": 1, "task_id": 10, "stage": "annotation", "state": "in progress", "organization": "ai-team"}

    def mock_urlopen(req, timeout=None):
        url = req.full_url
        method = req.get_method()
        calls.append((method, url))
        
        resp = MagicMock()
        resp.status = 200
        if "/api/jobs/1" in url and "/data/meta" not in url:
            if method == "PATCH":
                patch_data = json.loads(req.data.decode("utf-8"))
                current_job.update(patch_data)
                data = dict(current_job)
            else:
                data = dict(current_job)
        elif "/api/jobs/1/data/meta" in url:
            data = {"frames": []}
        elif "/api/issues" in url:
            data = {"results": []}
        else:
            data = {}

        resp.read.return_value = json.dumps(data).encode("utf-8")
        resp.__enter__.return_value = resp
        return resp

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    payload = {
        "cvat_url": "app.cvat.ai",  # Test URL without https://
        "cvat_access_token": "valid_token_123",
        "job_id": 1,
        "frame_offset": 0,
        "frame_map": {},
        "update_job_state": True,
        "target_stage": "validation",
        "target_state": "rejected",
        "sorted_frames": []  # Zero frames / zero issues return
    }

    req = MagicMock()
    req.headers = {}
    result = push_issues_to_cvat_payload(req, payload)

    assert result["status"] == "success"
    assert result["job_state_changed"] is True
    assert result["job_stage"] == "validation"
    assert result["job_state"] == "rejected"
    assert any(method == "PATCH" and "/api/jobs/1" in url for method, url in calls)


def test_return_job_fallback_plans(monkeypatch):
    calls = []
    current_job = {"id": 2, "task_id": 20, "stage": "annotation", "state": "in progress"}

    def mock_urlopen(req, timeout=None):
        url = req.full_url
        method = req.get_method()
        calls.append((method, url))
        
        resp = MagicMock()
        resp.status = 200
        if "/api/jobs/2" in url and "/data/meta" not in url:
            if method == "PATCH":
                patch_data = json.loads(req.data.decode("utf-8"))
                # Simulate CVAT rejecting annotation/rejected but accepting validation/rejected
                if patch_data.get("stage") == "annotation" and patch_data.get("state") == "rejected":
                    err = urllib.error.HTTPError(url, 400, "Invalid stage-state pair", {}, io.BytesIO(b'{"state":["rejected is invalid for annotation"]}'))
                    raise err
                current_job.update(patch_data)
                data = dict(current_job)
            else:
                data = dict(current_job)
        elif "/api/jobs/2/data/meta" in url:
            data = {"frames": [{"frame": 0, "name": "f0.jpg"}]}
        elif "/api/issues" in url:
            data = {"results": []}
        else:
            data = {}

        resp.read.return_value = json.dumps(data).encode("utf-8")
        resp.__enter__.return_value = resp
        return resp

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    payload = {
        "cvat_url": "https://app.cvat.ai",
        "cvat_access_token": "valid_token_123",
        "job_id": 2,
        "update_job_state": True,
        "target_stage": "annotation",
        "target_state": "rejected",
        "sorted_frames": [{"frame_name": "f0.jpg", "frame_index": 0, "errors": []}]
    }

    req = MagicMock()
    req.headers = {}
    result = push_issues_to_cvat_payload(req, payload)

    assert result["status"] == "success"
    assert result["job_state_changed"] is True
    # Successfully transitioned via fallback plan
    assert result["job_state"] == "rejected"

