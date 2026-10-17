"""
Modal Cloud GPU Vision Pipeline (YOLO11 + SAM 2.1 + DINOv2)
Cung cấp API Serverless chạy trên GPU Nvidia T4 (16GB VRAM) hoàn toàn miễn phí (Modal $30 free credits/month).
Tương thích 100% với QC Studio BDD100K 2D & Blazor Dashboard.

Cách deploy lên Modal:
1. pip install modal
2. python -m modal setup
3. modal deploy deploy_vision.py
"""
import io
import base64
import modal

# 1. Định nghĩa môi trường Linux + cài thư viện trên Cloud của Modal
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "fastapi[standard]",
        "python-multipart",
        "pillow",
        "numpy",
        "ultralytics>=8.3.0",
        "transformers",
    )
)

app = modal.App("vision-pipeline-api", image=image)

# Mapping COCO labels sang BDD100K ontology chuẩn của QC Studio
COCO_TO_BDD100K = {
    "person": "pedestrian",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "traffic light": "traffic light",
    "stop sign": "traffic sign",
    "train": "train",
}


# 2. Khởi tạo Class giữ 3 Model trên GPU Nvidia T4 (16GB VRAM)
@app.cls(gpu="T4", scaledown_window=180)  # Giữ GPU thức 3 phút sau mỗi lần gọi để liên tiếp cực nhanh và tiết kiệm chi phí
class VisionEngine:
    @modal.enter()
    def load_models(self):
        import torch
        from ultralytics import YOLO, SAM
        from transformers import AutoImageProcessor, AutoModel

        print("🚀 [Modal GPU] Đang tải 3 model vào VRAM của GPU Nvidia T4...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Load YOLO (Object Detection)
        self.yolo = YOLO("yolo11n.pt").to(self.device)
        print("✅ [Modal GPU] YOLO11 loaded successfully")

        # 2. Load SAM 2 / SAM 2.1 (Segment Anything)
        try:
            self.sam = SAM("sam2.1_b.pt").to(self.device)
            print("✅ [Modal GPU] SAM 2.1 loaded successfully")
        except Exception as exc:
            print(f"⚠️ [Modal GPU] sam2.1_b.pt fallback to sam2_b.pt: {exc}")
            self.sam = SAM("sam2_b.pt").to(self.device)

        # 3. Load DINOv2 (Feature Extraction / Foundation Model)
        dino_id = "facebook/dinov2-base"
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_id)
        self.dino_model = AutoModel.from_pretrained(dino_id).to(self.device).eval()
        print("✅ [Modal GPU] DINOv2 loaded successfully")
        print("🎉 [Modal GPU] Toàn bộ 3 Model đã sẵn sàng trên GPU T4!")

    @modal.method()
    def predict_full(self, image_bytes: bytes, threshold: float = 0.25):
        import torch
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        with torch.inference_mode():
            # A. Chạy YOLO Object Detection
            yolo_res = self.yolo(img, conf=threshold, verbose=False)
            boxes = []
            classes = []
            labels = []
            confidences = []
            predictions = []

            if yolo_res and len(yolo_res) > 0 and yolo_res[0].boxes is not None:
                for box in yolo_res[0].boxes:
                    xyxy = [float(c) for c in box.xyxy[0].tolist()]
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    raw_name = self.yolo.names.get(cls_id, str(cls_id))
                    bdd_name = COCO_TO_BDD100K.get(raw_name, raw_name)

                    boxes.append(xyxy)
                    classes.append(raw_name)
                    labels.append(bdd_name)
                    confidences.append(conf)

                    predictions.append({
                        "type": "rectangle",
                        "label": bdd_name,
                        "points": xyxy,
                        "confidence": conf
                    })

            # B. Chạy SAM 2.1 Segment Anything trên các bounding boxes
            segmentations = []
            if len(boxes) > 0:
                try:
                    sam_res = self.sam(img, bboxes=boxes, verbose=False)
                    if sam_res and len(sam_res) > 0 and sam_res[0].masks is not None:
                        for idx, seg in enumerate(sam_res[0].masks.xy):
                            if len(seg) >= 3:
                                flat_pts = [float(c) for pt in seg for c in pt]
                                lbl = labels[idx] if idx < len(labels) else "object"
                                cnf = confidences[idx] if idx < len(confidences) else 0.9
                                bx = boxes[idx] if idx < len(boxes) else []
                                segmentations.append({
                                    "type": "polygon",
                                    "label": lbl,
                                    "points": flat_pts,
                                    "box": bx,
                                    "confidence": cnf
                                })
                except Exception as ex:
                    print(f"⚠️ [Modal GPU] SAM prediction warning: {ex}")

            # C. Chạy DINOv2 trích xuất đặc trưng hình ảnh
            inputs = self.dino_processor(images=img, return_tensors="pt").to(self.device)
            feat = self.dino_model(**inputs).last_hidden_state[:, 0, :].squeeze().cpu().tolist()

        return {
            "num_detections": len(predictions),
            "boxes": boxes,
            "classes": classes,
            "labels": labels,
            "predictions": predictions,
            "masks": segmentations,
            "dino_embedding_preview": feat[:10]
        }


# 3. Tạo FastAPI Endpoint có bật CORS cho QC Studio & Blazor gọi vào
@app.function()
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, UploadFile, File, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    web_app = FastAPI(title="Modal GPU Vision API (YOLO + SAM 2.1 + DINOv2)")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    engine = VisionEngine()

    @web_app.get("/")
    @web_app.get("/status")
    def status():
        return {
            "status": "online",
            "gpu": "Nvidia T4 (16GB VRAM)",
            "model": "Modal Cloud GPU Ensemble (YOLO11 + SAM 2.1 + DINOv2)",
            "device": "cuda:0",
            "label_count": 80,
            "type": "ensemble"
        }

    @web_app.post("/")
    async def post_root(request: Request):
        try:
            body = await request.json()
            if body.get("action") == "status":
                return status()
        except Exception:
            pass
        return status()

    # 1. Endpoint test nguyên bản theo hướng dẫn của bạn (Hỗ trợ upload ảnh file hoặc json base64)
    @web_app.post("/analyze")
    async def analyze(file: UploadFile = File(None), request: Request = None):
        img_bytes = None
        if file is not None:
            img_bytes = await file.read()
        elif request is not None:
            try:
                body = await request.json()
                if "image" in body:
                    img_bytes = base64.b64decode(body["image"])
            except Exception:
                pass

        if not img_bytes:
            return JSONResponse(status_code=400, content={"error": "Vui lòng upload file ảnh hoặc gửi JSON có field 'image' (base64)"})

        res = engine.predict_full.remote(img_bytes)
        return res

    # 2. Endpoint chuẩn QC Studio cho Object Detection (BBox)
    @web_app.post("/bbox")
    async def bbox_infer(request: Request):
        try:
            body = await request.json()
            if body.get("action") == "status":
                return status()
            img_b64 = body.get("image", "")
            thresh = float(body.get("threshold", 0.25))
            img_bytes = base64.b64decode(img_b64)
            res = engine.predict_full.remote(img_bytes, threshold=thresh)
            return res.get("predictions", [])
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

    # 3. Endpoint chuẩn QC Studio cho Segmentation (SAM 2.1 / Polygon Masks)
    @web_app.post("/segmentation")
    @web_app.post("/sam2")
    async def seg_infer(request: Request):
        try:
            body = await request.json()
            if body.get("action") == "status":
                return status()
            img_b64 = body.get("image", "")
            thresh = float(body.get("threshold", 0.25))
            img_bytes = base64.b64decode(img_b64)
            res = engine.predict_full.remote(img_bytes, threshold=thresh)
            return res.get("masks", [])
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

    return web_app
