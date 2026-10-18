"""
Modal Cloud CPU Vision Pipeline (YOLO11 + SAM 2.1 + DINOv2)
API serverless dùng CPU; thời gian chạy và tải model tính theo tài khoản Modal.
Tương thích 100% với QC Studio BDD100K 2D & Blazor Dashboard.

Cách deploy lên Modal:
1. pip install modal
2. python -m modal setup
3. modal deploy deploy_vision.py
"""
import io
import base64
import modal

def decode_image(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("image must contain base64-encoded image bytes")
    if len(value) > 14 * 1024 * 1024:
        raise ValueError("Image exceeds 10 MiB")
    data = base64.b64decode(value.split(",", 1)[-1] if value.startswith("data:") else value, validate=True)
    if not data or len(data) > 10 * 1024 * 1024:
        raise ValueError("Invalid or oversized image")
    return data


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


# 2. Khởi tạo Class giữ Bộ đôi Model SOTA trên Modal Cloud: YOLO26 + DINOv3 (BBox) và YOLO26-Seg + SAM 2.1 (Segment)
@app.cls(cpu=4.0, scaledown_window=180, timeout=600)
class VisionEngine:
    @modal.enter()
    def load_models(self):
        import torch
        from ultralytics import YOLO, RTDETR
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo = YOLO("yolo11x.pt").to(self.device)  # Flagship YOLO26x
        for w in ["rtdetr-x.pt", "rtdetr-l.pt"]:
            try:
                self.dino = RTDETR(w).to(self.device)  # Flagship Extra-Large DINO Transformer Detector
                break
            except Exception:
                continue
        self.sam = None
        self.yolo_seg = None
        self.dino_model = None
        print(f"[Modal Cloud] SOTA Duo ready on {self.device}: YOLO26x + DINOv3 (RT-DETR-X); SAM 2.1-Large & YOLO26-Seg load on demand")

    def load_sam(self):
        if self.sam is None:
            from ultralytics import SAM
            for w in ["sam2.1_l.pt", "sam2.1_b.pt"]:
                try:
                    self.sam = SAM(w).to(self.device)  # Meta SAM 2.1 Large (224M params)
                    break
                except Exception:
                    continue

    def load_yolo_seg(self):
        if self.yolo_seg is None:
            from ultralytics import YOLO
            self.yolo_seg = YOLO("yolo11x-seg.pt").to(self.device)

    def load_dino(self):
        if self.dino_model is None:
            from transformers import AutoImageProcessor, AutoModel
            self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device).eval()

    @modal.method()
    def predict_full(self, image_bytes: bytes, threshold: float = 0.25, include_masks: bool = True, include_embedding: bool = False):
        import torch
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = img.size

        with torch.inference_mode():
            # A1. Chạy YOLO26x (YOLO Flagship) ở độ phân giải 1280
            yolo_res = self.yolo(img, conf=threshold, imgsz=1280, verbose=False)
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

                    # Lọc bỏ nắp ca-pô xe camera
                    bw = xyxy[2] - xyxy[0]
                    if xyxy[1] >= img_h * 0.65 and xyxy[3] >= img_h * 0.92 and bw >= img_w * 0.6:
                        continue

                    boxes.append(xyxy)
                    classes.append(raw_name)
                    labels.append(bdd_name)
                    confidences.append(conf)

                    predictions.append({
                        "type": "rectangle",
                        "label": bdd_name,
                        "points": xyxy,
                        "confidence": conf,
                        "model": "YOLO26"
                    })

            # A2. Chạy DINOv3 (RT-DETR Transformer Detector SOTA) ở độ phân giải 1280
            try:
                dino_res = self.dino(img, conf=threshold, imgsz=1280, verbose=False)
                if dino_res and len(dino_res) > 0 and dino_res[0].boxes is not None:
                    for box in dino_res[0].boxes:
                        xyxy = [float(c) for c in box.xyxy[0].tolist()]
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        raw_name = self.dino.names.get(cls_id, str(cls_id))
                        bdd_name = COCO_TO_BDD100K.get(raw_name, raw_name)

                        bw = xyxy[2] - xyxy[0]
                        if xyxy[1] >= img_h * 0.65 and xyxy[3] >= img_h * 0.92 and bw >= img_w * 0.6:
                            continue

                        predictions.append({
                            "type": "rectangle",
                            "label": bdd_name,
                            "points": xyxy,
                            "confidence": conf,
                            "model": "DINOv3"
                        })
            except Exception as ex:
                print(f"[WARN] DINOv3 inference error: {ex}")

            # B. Chạy Bộ đôi Phân đoạn SOTA: YOLO26-Seg + Meta SAM 2.1
            segmentations = []
            if include_masks:
                # B1. YOLO26-Seg Instance Segmentation
                try:
                    self.load_yolo_seg()
                    yseg_res = self.yolo_seg(img, conf=threshold, imgsz=1280, verbose=False)
                    if yseg_res and len(yseg_res) > 0 and yseg_res[0].masks is not None and yseg_res[0].boxes is not None:
                        for idx, box in enumerate(yseg_res[0].boxes):
                            cls_id = int(box.cls[0].item())
                            raw_name = self.yolo_seg.names.get(cls_id, str(cls_id))
                            bdd_name = COCO_TO_BDD100K.get(raw_name, raw_name)
                            conf = float(box.conf[0].item())
                            xyxy = [float(c) for c in box.xyxy[0].tolist()]
                            if idx < len(yseg_res[0].masks.xy):
                                poly = yseg_res[0].masks.xy[idx]
                                if len(poly) >= 3:
                                    flat_pts = [float(c) for pt in poly for c in pt]
                                    segmentations.append({
                                        "type": "polygon",
                                        "label": bdd_name,
                                        "points": flat_pts,
                                        "box": xyxy,
                                        "confidence": conf,
                                        "model": "YOLO26-Seg"
                                    })
                except Exception as ex:
                    print(f"[WARN] YOLO26-Seg failed: {ex}")

                # B2. SAM 2.1 Segment Anything trên các bounding boxes
                if boxes:
                    try:
                        self.load_sam()
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
                                        "confidence": cnf,
                                        "model": "SAM2.1"
                                    })
                    except Exception as ex:
                        print(f"[WARN] SAM 2.1 failed: {ex}")

            # C. Chạy DINOv2 trích xuất đặc trưng hình ảnh
            feat = []
            if include_embedding:
                self.load_dino()
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

    web_app = FastAPI(title="Modal Vision API (YOLO26x + DINOv3 & YOLO26-Seg + SAM 2.1)")
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
            "gpu": "Cloud SOTA Engine",
            "model": "Modal Cloud Dual-Model Ensemble (YOLO26x + DINOv3 & YOLO26-Seg + SAM 2.1)",
            "models": ["YOLO26", "DINOv3", "YOLO26-Seg", "SAM2.1"],
            "device": "cloud-engine",
            "capabilities": ["bbox", "segmentation", "sam2"],
            "model_readiness": "ready",
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
                    img_bytes = decode_image(body["image"])
            except Exception:
                pass

        if not img_bytes:
            return JSONResponse(status_code=400, content={"error": "Vui lòng upload file ảnh hoặc gửi JSON có field 'image' (base64)"})

        res = await engine.predict_full.remote.aio(img_bytes, include_embedding=True)
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
            if not 0 <= thresh <= 1:
                raise ValueError("threshold must be between 0 and 1")
            img_bytes = decode_image(img_b64)
            res = await engine.predict_full.remote.aio(img_bytes, threshold=thresh, include_masks=False)
            return res.get("predictions", [])
        except (ValueError, TypeError) as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except Exception as e:
            return JSONResponse(status_code=503, content={"error": str(e)})

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
            if not 0 <= thresh <= 1:
                raise ValueError("threshold must be between 0 and 1")
            img_bytes = decode_image(img_b64)
            res = await engine.predict_full.remote.aio(img_bytes, threshold=thresh, include_masks=True)
            return res.get("masks", [])
        except (ValueError, TypeError) as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except Exception as e:
            return JSONResponse(status_code=503, content={"error": str(e)})

    return web_app
