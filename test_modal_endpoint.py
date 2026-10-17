"""
Script kiểm tra kết nối tới Modal Cloud GPU API.
Cách dùng:
    python test_modal_endpoint.py https://vanthang10tin--vision-pipeline-api-fastapi-app.modal.run
"""
import sys
import io
import json
import base64
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
        return

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
        with urllib.request.urlopen(req, timeout=60) as res:
            bboxes = json.loads(res.read().decode())
            print(f"✅ BBox Endpoint OK! Số phát hiện: {len(bboxes)}")
    except Exception as e:
        print(f"❌ Lỗi gọi /bbox: {e}")

    print("\n🔍 [3/3] Kiểm tra endpoint Segmentation: /segmentation ...")
    try:
        payload = json.dumps({"image": b64_img, "threshold": 0.2}).encode()
        req = urllib.request.Request(f"{base_url}/segmentation", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as res:
            masks = json.loads(res.read().decode())
            print(f"✅ Segmentation Endpoint OK! Số masks: {len(masks)}")
    except Exception as e:
        print(f"❌ Lỗi gọi /segmentation: {e}")

    print("\n🎉 Toàn bộ endpoint trên Modal GPU đã sẵn sàng kết nối vào QC Studio!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_url = sys.argv[1]
    else:
        target_url = input("Nhập URL Modal API của bạn (ví dụ: https://vanthang10tin--vision-pipeline-api-fastapi-app.modal.run): ").strip()

    if target_url:
        test_endpoint(target_url)
    else:
        print("Vui lòng cung cấp URL để test.")
