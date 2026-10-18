"""
Script kiểm tra kết nối tới Modal Cloud GPU API.
Cách dùng:
    python test_modal_endpoint.py https://<your-modal-app>.modal.run
Hoặc đặt QC_MODAL_API_URL trong môi trường rồi chạy script không tham số.
"""
import argparse
import io
import json
import base64
import os
import urllib.request
from PIL import Image, ImageDraw

def test_endpoint(base_url: str):
    base_url = base_url.strip().rstrip("/")
    print(f"\n🔍 [1/3] Kiểm tra trạng thái máy chủ GPU: {base_url}/status ...")
    try:
        req = urllib.request.Request(f"{base_url}/status", headers={"User-Agent": "QC-Studio-Tester"})
        with urllib.request.urlopen(req, timeout=30) as res:
            info = json.loads(res.read().decode())
            print(f"✅ Máy chủ phản hồi OK: {info}")
    except Exception as e:
        print(f"❌ Không kết nối được đến status endpoint: {e}")
        return False

    failures = []

    # Tạo 1 ảnh mẫu đơn giản để test inference
    img = Image.new("RGB", (320, 240), color=(70, 70, 70))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 180, 160], fill=(200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64_img = base64.b64encode(buf.getvalue()).decode()

    print("\n🔍 [2/3] Kiểm tra endpoint Object Detection (BBox): /bbox ...")
    try:
        payload = json.dumps({"image": b64_img, "threshold": 0.2}).encode()
        req = urllib.request.Request(f"{base_url}/bbox", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as res:
            bboxes = json.loads(res.read().decode())
            if not isinstance(bboxes, list):
                raise ValueError("BBox response must be a list")
            print(f"✅ BBox Endpoint OK! Số phát hiện: {len(bboxes)}")
    except Exception as e:
        failures.append("bbox")
        print(f"❌ Lỗi gọi /bbox: {e}")

    print("\n🔍 [3/3] Kiểm tra endpoint Segmentation: /segmentation ...")
    try:
        payload = json.dumps({"image": b64_img, "threshold": 0.2}).encode()
        req = urllib.request.Request(f"{base_url}/segmentation", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as res:
            masks = json.loads(res.read().decode())
            if not isinstance(masks, list):
                raise ValueError("Segmentation response must be a list")
            print(f"✅ Segmentation Endpoint OK! Số masks: {len(masks)}")
    except Exception as e:
        failures.append("segmentation")
        print(f"❌ Lỗi gọi /segmentation: {e}")

    if failures:
        return False
    print("\n🎉 Toàn bộ endpoint trên Modal GPU đã sẵn sàng kết nối vào QC Studio!")

    return True

test_endpoint.__test__ = False  # CLI smoke check, not a pytest fixture-based test.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check a configured Modal vision API endpoint.")
    parser.add_argument(
        "url",
        nargs="?",
        default=os.environ.get("QC_MODAL_API_URL", ""),
        help="Modal API base URL (defaults to QC_MODAL_API_URL)",
    )
    args = parser.parse_args()
    if not args.url.strip():
        parser.error("provide a URL or set QC_MODAL_API_URL")
    raise SystemExit(0 if test_endpoint(args.url) else 1)
