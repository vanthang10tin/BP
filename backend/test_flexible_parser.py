import io
import zipfile

import pytest
import flexible_dataset_parser as parser
from flexible_dataset_parser import parse_flexible_dataset


def test_parse_bdd_json():
    content = b'[{"name": "frame_1.jpg", "labels": [{"category": "car", "box2d": {"x1": 10, "y1": 20, "x2": 100, "y2": 200}}]}]'
    res = parse_flexible_dataset(content, "dataset.json")
    assert res["status"] == "success"
    assert res["format"] == "bdd100k_json"
    assert res["frame_count"] == 1
    assert res["frames"][0]["name"] == "frame_1.jpg"
    assert res["frames"][0]["labels"][0]["category"] == "car"


def test_parse_voc_xml():
    xml = b"""<annotation>
        <filename>000001.jpg</filename>
        <size><width>1920</width><height>1080</height></size>
        <object>
            <name>truck</name>
            <bndbox><xmin>100</xmin><ymin>200</ymin><xmax>500</xmax><ymax>600</ymax></bndbox>
        </object>
    </annotation>"""
    res = parse_flexible_dataset(xml, "000001.xml")
    assert res["status"] == "success"
    assert res["format"] == "pascal_voc_xml"
    assert res["frame_count"] == 1
    assert res["frames"][0]["name"] == "000001.jpg"
    assert res["frames"][0]["labels"][0]["category"] == "truck"
    assert res["frames"][0]["labels"][0]["box2d"]["x1"] == 100.0


def test_parse_cvat_xml():
    xml = b"""<annotations>
        <image id="0" name="test.jpg" width="1280" height="720">
            <box label="bus" xtl="50" ytl="60" xbr="400" ybr="300" occluded="1" />
        </image>
    </annotations>"""
    res = parse_flexible_dataset(xml, "annotations.xml")
    assert res["status"] == "success"
    assert res["format"] == "cvat_xml"
    assert res["frame_count"] == 1
    assert res["frames"][0]["labels"][0]["category"] == "bus"
    assert res["frames"][0]["labels"][0]["attributes"]["occluded"] is True


def test_parse_csv():
    csv_data = b"filename,category,xmin,ymin,xmax,ymax\nimg1.jpg,pedestrian,20,30,80,150\n"
    res = parse_flexible_dataset(csv_data, "labels.csv")
    assert res["status"] == "success"
    assert res["format"] == "csv_table"
    assert res["frame_count"] == 1
    assert res["frames"][0]["name"] == "img1.jpg"
    assert res["frames"][0]["labels"][0]["category"] == "pedestrian"


def test_parse_yolo_txt():
    txt = b"0 0.5 0.5 0.2 0.4\n"
    res = parse_flexible_dataset(txt, "frame_01.txt", profile={"classes": ["car"]})
    assert res["status"] == "success"
    assert res["format"] == "yolo_txt"
    assert res["frame_count"] == 1
    assert res["frames"][0]["name"] == "frame_01.jpg"
    assert res["frames"][0]["labels"][0]["category"] == "car"


def test_zip_rejects_too_many_members(monkeypatch):
    monkeypatch.setattr(parser, "MAX_ARCHIVE_ENTRIES", 1)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("one.txt", "")
        zf.writestr("two.txt", "")
    with pytest.raises(ValueError, match="quá nhiều file"):
        parse_flexible_dataset(archive.getvalue(), "dataset.zip")


def test_zip_rejects_excessive_uncompressed_size(monkeypatch):
    monkeypatch.setattr(parser, "MAX_ARCHIVE_UNCOMPRESSED_BYTES", 1)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("labels.csv", "a,b")
    with pytest.raises(ValueError, match="sau giải nén"):
        parse_flexible_dataset(archive.getvalue(), "dataset.zip")
