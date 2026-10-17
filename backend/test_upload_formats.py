#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kiểm thử tự động các định dạng nạp dữ liệu: VOC XML, YOLO TXT, CSV và ZIP Archive.
Hoạt động độc lập qua FastAPI TestClient cho pytest và CLI.
"""

import os
import sys
import io
import zipfile
import pytest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from app import fastapi_app
import flexible_dataset_parser
from fastapi.testclient import TestClient

client = TestClient(fastapi_app)

VOC_XML = b"""<annotation>
    <filename>voc_car.jpg</filename>
    <size><width>1280</width><height>720</height><depth>3</depth></size>
    <object><name>car</name><bndbox><xmin>100</xmin><ymin>150</ymin><xmax>400</xmax><ymax>350</ymax></bndbox></object>
</annotation>"""

YOLO_TXT = b"0 0.5 0.5 0.2 0.3\n1 0.2 0.3 0.1 0.1\n"

CSV_DATA = b"filename,class,x1,y1,x2,y2\nframe_01.jpg,pedestrian,50,100,80,220\nframe_02.jpg,truck,200,300,500,600\n"


def test_upload_voc_xml():
    res = client.post("/api/qc/check", files={"file": ("voc_car.xml", VOC_XML, "text/xml")})
    assert res.status_code == 200, f"Failed: {res.text}"
    data = res.json()
    assert data.get("status") == "success"
    assert len(data.get("sorted_frames", [])) >= 1


def test_upload_yolo_txt():
    res = client.post("/api/qc/check", files={"file": ("frame_0001.txt", YOLO_TXT, "text/plain")})
    assert res.status_code == 200, f"Failed: {res.text}"
    data = res.json()
    assert data.get("status") == "success"
    assert len(data.get("sorted_frames", [])) >= 1


def test_upload_csv():
    res = client.post("/api/qc/check", files={"file": ("labels.csv", CSV_DATA, "text/csv")})
    assert res.status_code == 200, f"Failed: {res.text}"
    data = res.json()
    assert data.get("status") == "success"
    assert len(data.get("sorted_frames", [])) >= 1


def test_upload_zip_archive():
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("annotations/test.xml", VOC_XML)
        zf.writestr("images/voc_car.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 20)
    res = client.post("/api/qc/check", files={"file": ("dataset.zip", zip_buf.getvalue(), "application/zip")})
    assert res.status_code == 200, f"Failed: {res.text}"
    data = res.json()
    assert data.get("status") == "success"
    assert len(data.get("sorted_frames", [])) >= 1


def test_flexible_upload_rejects_payload_over_limit(monkeypatch):
    monkeypatch.setattr(flexible_dataset_parser, "MAX_UPLOAD_BYTES", 4)
    res = client.post(
        "/api/dataset/parse-flexible",
        files={"file": ("labels.csv", b"12345", "text/csv")},
    )
    assert res.status_code == 413
    assert "64 MiB" in res.json()["detail"]


if __name__ == "__main__":
    print("Testing upload formats with TestClient...")
    test_upload_voc_xml()
    print("[1. XML Upload] PASSED")
    test_upload_yolo_txt()
    print("[2. YOLO TXT Upload] PASSED")
    test_upload_csv()
    print("[3. CSV Upload] PASSED")
    test_upload_zip_archive()
    print("[4. ZIP Upload] PASSED")
    print("ALL FORMATS VERIFIED WORKING!")
