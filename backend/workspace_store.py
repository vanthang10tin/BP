"""Small local SQLite registry for reusable dataset profiles and guideline history."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(os.environ.get("QC_DATA_DIR", Path(__file__).resolve().parents[1] / ".qc-data")) / "qc_workspaces.sqlite3"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS workspaces (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, task_type TEXT NOT NULL,
            annotation_format TEXT NOT NULL, profile_json TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS guideline_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, workspace_id TEXT NOT NULL,
            version INTEGER NOT NULL, text TEXT NOT NULL, sha256 TEXT NOT NULL,
            status TEXT NOT NULL, created_at TEXT NOT NULL,
            UNIQUE(workspace_id, version),
            FOREIGN KEY(workspace_id) REFERENCES workspaces(id)
        );
    """)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_workspace(name: str, profile: dict[str, Any]) -> dict[str, Any]:
    name = re.sub(r"\s+", " ", str(name)).strip()[:120]
    task = profile.get("task_type", "bbox")
    fmt = profile.get("annotation_format", "bdd100k")
    VALID_FORMATS = {"bdd100k", "coco", "yolo_json", "normalized", "voc", "cvat_xml", "yolo_txt", "csv", "zip", "custom", "flexible", "pascal_voc_xml"}
    if not name or task not in {"bbox", "semantic_segmentation"}:
        raise ValueError("Workspace needs a name and task_type bbox or semantic_segmentation")
    if fmt not in VALID_FORMATS:
        fmt = "normalized"
    workspace_id = str(uuid.uuid4())
    now = _now()
    conn = _connect()
    try:
        conn.execute("INSERT INTO workspaces VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (workspace_id, name, task, fmt, json.dumps(profile, ensure_ascii=False), now, now))
        conn.commit()
    finally:
        conn.close()
    return get_workspace(workspace_id)


def get_workspace(workspace_id: str) -> dict[str, Any] | None:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM workspaces WHERE id=?", (workspace_id,)).fetchone()
        if not row:
            return None
        obj = dict(row)
        obj["profile"] = json.loads(obj.pop("profile_json"))
        obj["guidelines"] = list_guidelines(workspace_id, conn=conn)
        obj["active_guideline"] = next((g for g in reversed(obj["guidelines"]) if g["status"] == "active"), None)
        return obj
    finally:
        conn.close()


def list_workspaces() -> list[dict[str, Any]]:
    conn = _connect()
    try:
        ids = [r[0] for r in conn.execute("SELECT id FROM workspaces ORDER BY updated_at DESC")]
    finally:
        conn.close()
    workspaces = [get_workspace(i) for i in ids]
    return [workspace for workspace in workspaces if workspace]


def update_profile(workspace_id: str, profile: dict[str, Any]) -> dict[str, Any] | None:
    current = get_workspace(workspace_id)
    if not current:
        return None
    task = profile.get("task_type", current["task_type"])
    fmt = profile.get("annotation_format", current["annotation_format"])
    VALID_FORMATS = {"bdd100k", "coco", "yolo_json", "normalized", "voc", "cvat_xml", "yolo_txt", "csv", "zip", "custom", "flexible", "pascal_voc_xml"}
    if task not in {"bbox", "semantic_segmentation"}:
        raise ValueError("Invalid task_type")
    if fmt not in VALID_FORMATS:
        fmt = "normalized"
    conn = _connect()
    try:
        conn.execute("UPDATE workspaces SET task_type=?, annotation_format=?, profile_json=?, updated_at=? WHERE id=?",
                     (task, fmt, json.dumps(profile, ensure_ascii=False), _now(), workspace_id))
        conn.commit()
    finally:
        conn.close()
    return get_workspace(workspace_id)


def list_guidelines(workspace_id: str, conn: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
    own = conn is None
    conn = conn or _connect()
    try:
        rows = conn.execute("SELECT version, sha256, status, created_at, text FROM guideline_versions WHERE workspace_id=? ORDER BY version", (workspace_id,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        if own:
            conn.close()


def save_guideline(workspace_id: str, text: str, status: str = "draft") -> dict[str, Any]:
    if status not in {"draft", "active"} or not text.strip():
        raise ValueError("Guideline text is required and status must be draft or active")
    conn = _connect()
    try:
        if not conn.execute("SELECT 1 FROM workspaces WHERE id=?", (workspace_id,)).fetchone():
            raise KeyError(workspace_id)
        if status == "active":
            conn.execute("UPDATE guideline_versions SET status='archived' WHERE workspace_id=? AND status='active'", (workspace_id,))
        version = conn.execute("SELECT COALESCE(MAX(version),0)+1 FROM guideline_versions WHERE workspace_id=?", (workspace_id,)).fetchone()[0]
        item = {"version": version, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "status": status, "created_at": _now(), "text": text}
        conn.execute("INSERT INTO guideline_versions (workspace_id,version,text,sha256,status,created_at) VALUES (?,?,?,?,?,?)",
                     (workspace_id, version, text, item["sha256"], status, item["created_at"]))
        conn.execute("UPDATE workspaces SET updated_at=? WHERE id=?", (_now(), workspace_id))
        conn.commit()
        return item
    finally:
        conn.close()


def activate_guideline(workspace_id: str, version: int) -> dict[str, Any] | None:
    conn = _connect()
    try:
        if not conn.execute("SELECT 1 FROM guideline_versions WHERE workspace_id=? AND version=?", (workspace_id, version)).fetchone():
            return None
        conn.execute("UPDATE guideline_versions SET status=CASE WHEN version=? THEN 'active' ELSE 'archived' END WHERE workspace_id=?", (version, workspace_id))
        conn.execute("UPDATE workspaces SET updated_at=? WHERE id=?", (_now(), workspace_id))
        conn.commit()
    finally:
        conn.close()
    return next((item for item in list_guidelines(workspace_id) if item["version"] == version), None)
