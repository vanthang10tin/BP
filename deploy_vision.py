"""
Modal Cloud Vision Pipeline (YOLO26x + D-FINE-X & YOLO26x-Seg + SAM 2.1 Large)
API serverless chạy 100% trên GPU Nvidia T4 Cloud.
- BBox: YOLO26x (Ultralytics End-to-End NMS-Free) + D-FINE-X (Peterande/D-FINE, arXiv:2410.13842)
- Segmentation: YOLO26x-Seg (Ultralytics) + Meta SAM 2.1 Large (Hiera)
Tương thích 100% với QC Studio BDD100K 2D & Blazor Dashboard.
"""
import io
import base64
import os
import modal

def decode_image(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("image must contain base64-encoded image bytes")
    if len(value) > 14 * 1024 * 1024:
        raise ValueError("Image exceeds 10 MiB")
    data = base64.b64decode(value.split(",", 1)[-1] if value.startswith("data:") else value, validate=True)
    if not data or len(data) > 10 * 1024 * 1024:
        raise ValueError("Invalid or oversized image")
    validate_image(data)
    return data


def validate_image(data: bytes):
    from PIL import Image
    if not data or len(data) > 10 * 1024 * 1024:
        raise ValueError("Invalid or oversized image")
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.width * img.height > 25_000_000:
                raise ValueError("Image exceeds 25 megapixels")
            img.verify()
    except Exception as exc:
        raise ValueError("Invalid or oversized image") from exc


def map_predictions(predictions, body):
    mapping = body.get("mapping") or {}
    if not isinstance(mapping, dict):
        raise ValueError("mapping must be an object")
    task_labels = body.get("task_labels") or []
    defaults = {"pedestrian": "person"} if "person" in task_labels and "pedestrian" not in task_labels else {}
    effective = {**defaults, **mapping}
    return [{**item, "label": effective.get(item["label"], item["label"])} for item in predictions]


modal_runtime_env = {
    key: os.environ[key]
    for key in ("QC_MODAL_GPU", "QC_MODAL_YOLO_WEIGHTS", "QC_MODAL_DFINE_WEIGHTS", "QC_MODAL_YOLO_SEG_WEIGHTS", "QC_MODAL_SAM_WEIGHTS")
    if os.environ.get(key)
}

# 1. Image container với weights YOLO26x, D-FINE-X, YOLO26x-Seg và SAM 2.1 Large tải sẵn
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0", "curl")
    .pip_install(
        "torch",
        "torchvision",
        "fastapi[standard]",
        "python-multipart",
        "pillow",
        "numpy",
        "ultralytics>=8.4.0",
        "transformers",
        "scipy",
        "pyyaml",
        "tensorboard",
        "loguru",
        "faster-coco-eval",
        "calflops",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/Peterande/D-FINE.git /root/D-FINE",
        "curl --fail --location --retry 3 -o /root/dfine_x_coco.pth https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_coco.pth",
        "curl --fail --location --retry 3 -o /root/yolo26x.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt",
        "curl --fail --location --retry 3 -o /root/yolo26x-seg.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-seg.pt",
        "curl --fail --location --retry 3 -o /root/sam2.1_l.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_l.pt",
    )
    .env(modal_runtime_env)
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

COCO_CLASSES_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


# 2. Khởi tạo VisionEngine chạy trên GPU Nvidia T4
@app.cls(gpu=os.environ.get("QC_MODAL_GPU", "T4").strip() or None,
         scaledown_window=300, timeout=600, max_containers=10)
class VisionEngine:
    # Modal creates separate container pools for bbox and segmentation.
    pipeline: str = modal.parameter(default="bbox")

    @modal.enter()
    def load_models(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bbox_loaded = False
        self.segmentation_loaded = False
        self.dino_model = None

    def load_bbox_models(self):
        if self.bbox_loaded:
            return
        import sys
        if "/root/D-FINE" not in sys.path:
            sys.path.insert(0, "/root/D-FINE")
        import torch
        import torch.nn as nn
        import torchvision.transforms as T
        from ultralytics import YOLO

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_weights = os.environ.get("QC_MODAL_YOLO_WEIGHTS", "/root/yolo26x.pt")
        self.dfine_weights = os.environ.get("QC_MODAL_DFINE_WEIGHTS", "/root/dfine_x_coco.pth")
        # 1. Flagship One-Stage: YOLO26x
        self.yolo = YOLO(self.yolo_weights).to(self.device)

        # 2. Flagship DETR: D-FINE-X (Peterande/D-FINE, arXiv:2410.13842)
        try:
            from src.core import YAMLConfig
            dfine_cfg_path = "/root/D-FINE/configs/dfine/dfine_hgnetv2_x_coco.yml"
            dfine_cfg = YAMLConfig(dfine_cfg_path, resume=self.dfine_weights)
            if "HGNetv2" in dfine_cfg.yaml_cfg:
                dfine_cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

            dfine_ckpt = torch.load(self.dfine_weights, map_location="cpu")
            dfine_state = dfine_ckpt["ema"]["module"] if "ema" in dfine_ckpt else dfine_ckpt["model"]
            dfine_cfg.model.load_state_dict(dfine_state)

            class DFineDeployModel(nn.Module):
                def __init__(self, m, p):
                    super().__init__()
                    self.model = m.deploy()
                    self.postprocessor = p.deploy()

                def forward(self, images, orig_target_sizes):
                    outputs = self.model(images)
                    return self.postprocessor(outputs, orig_target_sizes)

            self.dfine = DFineDeployModel(dfine_cfg.model, dfine_cfg.postprocessor).to(self.device).eval()
            self.dfine_transforms = T.Compose([
                T.Resize((640, 640)),
                T.ToTensor(),
            ])
            print(f"[Modal Cloud GPU] D-FINE-X loaded successfully on {self.device}")
        except Exception as e:
            print(f"[Modal Cloud GPU] Failed to load D-FINE-X: {e}")
            self.dfine = None

        self.bbox_loaded = True

    def load_segmentation_models(self):
        if self.segmentation_loaded:
            return
        from ultralytics import YOLO, SAM
        self.yolo_seg_weights = os.environ.get("QC_MODAL_YOLO_SEG_WEIGHTS", "/root/yolo26x-seg.pt")
        self.sam_weights = os.environ.get("QC_MODAL_SAM_WEIGHTS", "/root/sam2.1_l.pt")
        # 3. Flagship Segmentation: YOLO26x-Seg
        self.yolo_seg = YOLO(self.yolo_seg_weights).to(self.device)

        # 4. Flagship Refiner: Meta SAM 2.1 Large
        self.sam = SAM(self.sam_weights).to(self.device)

        # 5. Semantic Backbone: SegFormer Cityscapes (19 classes: Road, Sidewalk, Building, Sky, Vegetation...)
        try:
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            self.seg_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
            self.seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024").to(self.device).eval()
            print(f"[Modal Cloud GPU] SegFormer Cityscapes loaded on {self.device} for 19-class semantic coverage")
        except Exception as e:
            print(f"[Modal Cloud GPU] SegFormer load warning: {e}")
            raise RuntimeError("Semantic segmentation backbone unavailable") from e

        self.segmentation_loaded = True

    def load_dino(self):
        if self.dino_model is None:
            from transformers import AutoImageProcessor, AutoModel
            self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device).eval()

    # Luồng Object Detection độc lập: YOLO26x + D-FINE-X
    @modal.method()
    def predict_bbox(self, image_bytes: bytes, threshold: float = 0.25):
        import torch
        from PIL import Image

        self.load_bbox_models()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        with torch.inference_mode():
            # A1. YOLO26x Flagship Detector
            yolo_res = self.yolo(img, conf=threshold, imgsz=640, verbose=False)
            model_runs = [{"role": "detector_a", "name": "yolo26x.pt", "status": "success"}]
            predictions = []

            if yolo_res and len(yolo_res) > 0 and yolo_res[0].boxes is not None:
                for box in yolo_res[0].boxes:
                    xyxy = [float(c) for c in box.xyxy[0].tolist()]
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    raw_name = self.yolo.names.get(cls_id, str(cls_id))
                    bdd_name = COCO_TO_BDD100K.get(raw_name, raw_name)
                    predictions.append({
                        "type": "rectangle",
                        "label": bdd_name,
                        "points": xyxy,
                        "confidence": conf,
                        "model": "yolo26x.pt",
                        "model_role": "detector_a"
                    })

            if self.dfine is None:
                model_runs.append({"role": "detector_b", "name": "dfine_x_coco.pth", "status": "failed"})

            # A2. D-FINE-X Fine-Grained Distribution Refinement DETR
            if self.dfine is not None:
                try:
                    im_data = self.dfine_transforms(img).unsqueeze(0).to(self.device)
                    orig_size = torch.tensor([[w, h]], device=self.device)
                    dfine_outputs = self.dfine(im_data, orig_size)
                    labels, boxes, scores = dfine_outputs
                    model_runs.append({"role": "detector_b", "name": "dfine_x_coco.pth", "status": "success"})

                    scr = scores[0]
                    mask = scr >= threshold
                    lab = labels[0][mask]
                    bx = boxes[0][mask]
                    sc = scr[mask]

                    for j in range(len(sc)):
                        cls_id = int(lab[j].item())
                        raw_name = COCO_CLASSES_80[cls_id] if 0 <= cls_id < len(COCO_CLASSES_80) else str(cls_id)
                        bdd_name = COCO_TO_BDD100K.get(raw_name, raw_name)
                        xyxy = [round(float(c), 2) for c in bx[j].tolist()]
                        conf = round(float(sc[j].item()), 4)
                        predictions.append({
                            "type": "rectangle",
                            "label": bdd_name,
                            "points": xyxy,
                            "confidence": conf,
                            "model": "dfine_x_coco.pth",
                            "model_role": "detector_b"
                        })
                except Exception as ex:
                    model_runs = [run for run in model_runs if run["role"] != "detector_b"]
                    model_runs.append({"role": "detector_b", "name": "dfine_x_coco.pth", "status": "failed"})
                    print(f"[WARN] D-FINE inference error: {ex}")

        supported_classes = sorted({COCO_TO_BDD100K.get(n, n) for n in self.yolo.names.values()})
        for run in model_runs:
            run["supported_classes"] = supported_classes

        return {
            "predictions": predictions,
            "model_runs": model_runs,
            "num_detections": len(predictions)
        }

    # Luồng Segmentation độc lập: SegFormer (Dense 19-class Background) + YOLO26x-Seg + Meta SAM 2.1 Large
    @modal.method()
    def predict_segmentation(self, image_bytes: bytes, threshold: float = 0.25):
        import torch
        import numpy as np
        from PIL import Image

        self.load_segmentation_models()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        segmentations = []
        seg_boxes = []
        seg_labels = []
        seg_confs = []

        with torch.inference_mode():
            # B0. SegFormer Cityscapes Semantic Backbone cho các lớp background (đường, trời, nhà, cây, vỉa hè...)
            if getattr(self, "seg_model", None) is not None:
                try:
                    inputs = self.seg_processor(images=img, return_tensors="pt").to(self.device)
                    outputs = self.seg_model(**inputs)
                    logits = torch.nn.functional.interpolate(
                        outputs.logits, size=(h, w), mode="bilinear", align_corners=False
                    )
                    probabilities = logits.softmax(dim=1)[0]
                    pred_labels = probabilities.argmax(dim=0).cpu().numpy()
                    
                    # Full Cityscapes ontology, including rider and non-stop traffic signs.
                    SEMANTIC_CLASSES = {
                        0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
                        5: "pole", 6: "traffic light", 7: "traffic sign",
                        8: "vegetation", 9: "terrain", 10: "sky", 11: "pedestrian",
                        12: "rider", 13: "car", 14: "truck", 15: "bus",
                        16: "train", 17: "motorcycle", 18: "bicycle"
                    }
                    for cls_id, cls_name in SEMANTIC_CLASSES.items():
                        bin_mask = (pred_labels == cls_id)
                        if bin_mask.any():
                            y_idx, x_idx = np.where(bin_mask)
                            if len(x_idx) > 0:
                                x1, y1, x2, y2 = int(x_idx.min()), int(y_idx.min()), int(x_idx.max()), int(y_idx.max())
                                roi = bin_mask[y1:y2 + 1, x1:x2 + 1].astype(np.uint8).flatten().tolist()
                                segmentations.append({
                                    "type": "mask",
                                    "label": cls_name,
                                    "mask": roi + [x1, y1, x2, y2],
                                    "box": [x1, y1, x2, y2],
                                    "confidence": float(probabilities[cls_id][torch.as_tensor(bin_mask, device=self.device)].mean().item()),
                                    "model": "segformer-b0"
                                })
                except Exception as ex:
                    raise RuntimeError("Semantic segmentation inference failed") from ex

            # B1. YOLO26x-Seg Flagship Instance Segmentation (Phân giải cao 1280px phát hiện vật thể nhỏ/xa)
            try:
                yseg_res = self.yolo_seg(img, conf=threshold, imgsz=1280, verbose=False)
                if yseg_res and len(yseg_res) > 0 and yseg_res[0].masks is not None and yseg_res[0].boxes is not None:
                    for idx, box in enumerate(yseg_res[0].boxes):
                        cls_id = int(box.cls[0].item())
                        raw_name = self.yolo_seg.names.get(cls_id, str(cls_id))
                        bdd_name = COCO_TO_BDD100K.get(raw_name, raw_name)
                        conf = float(box.conf[0].item())
                        xyxy = [float(c) for c in box.xyxy[0].tolist()]
                        seg_boxes.append(xyxy)
                        seg_labels.append(bdd_name)
                        seg_confs.append(conf)
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
                                    "model": "yolo26x-seg.pt"
                                })
            except Exception as ex:
                raise RuntimeError("Instance segmentation inference failed") from ex

            # B2. SAM 2.1 Large Segment Anything trên TẤT CẢ bounding boxes (Không cắt giới hạn :10)
            if seg_boxes:
                try:
                    for start in range(0, len(seg_boxes), 16):
                        sam_res = self.sam(img, bboxes=seg_boxes[start:start + 16], verbose=False)
                        if sam_res and len(sam_res) > 0 and sam_res[0].masks is not None:
                            for offset, seg in enumerate(sam_res[0].masks.xy):
                                idx = start + offset
                                if len(seg) >= 3:
                                    flat_pts = [float(c) for pt in seg for c in pt]
                                    lbl = seg_labels[idx] if idx < len(seg_labels) else "object"
                                    cnf = seg_confs[idx] if idx < len(seg_confs) else 0.95
                                    bx = seg_boxes[idx] if idx < len(seg_boxes) else []
                                    segmentations.append({
                                        "type": "polygon",
                                        "label": lbl,
                                        "points": flat_pts,
                                        "box": bx,
                                        "confidence": cnf,
                                        "model": "sam2.1_l.pt"
                                    })
                except Exception as ex:
                    raise RuntimeError("SAM refinement failed") from ex

        return segmentations

    # Phân tích tổng thể
    @modal.method()
    def predict_full(self, image_bytes: bytes, threshold: float = 0.25, include_masks: bool = True, include_embedding: bool = False):
        bbox_res = self.predict_bbox.local(image_bytes, threshold=threshold)
        segmentations = self.predict_segmentation.local(image_bytes, threshold=threshold) if include_masks else []

        feat = []
        if include_embedding:
            import torch
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            self.load_dino()
            with torch.inference_mode():
                inputs = self.dino_processor(images=img, return_tensors="pt").to(self.device)
                feat = self.dino_model(**inputs).last_hidden_state[:, 0, :].squeeze().cpu().tolist()

        return {
            "num_detections": bbox_res.get("num_detections", 0),
            "predictions": bbox_res.get("predictions", []),
            "model_runs": bbox_res.get("model_runs", []),
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

    web_app = FastAPI(title="Modal Vision API (YOLO26x + D-FINE-X & YOLO26x-Seg + SAM 2.1 Large)")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    engine = VisionEngine()
    segmentation_engine = VisionEngine(pipeline="segmentation")

    @web_app.get("/")
    @web_app.get("/status")
    def status():
        return {
            "status": "online",
            "gpu": os.environ.get("QC_MODAL_GPU", "T4").strip() or None,
            "model": "Modal Dual-Detector (YOLO26x + D-FINE-X) & Flagship Seg (YOLO26x-Seg + SAM 2.1 Large)",
            "models": ["yolo26x.pt", "dfine_x_coco.pth", "yolo26x-seg.pt", "sam2.1_l.pt"],
            "device": "unknown",
            "capabilities": ["bbox", "segmentation", "sam2"],
            "model_readiness": "not_checked",
            "label_count": 80,
            "type": "ensemble"
        }

    @web_app.post("/")
    async def post_root(request: Request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("JSON body must be an object")
            if body.get("action") == "status":
                return status()
        except Exception:
            pass
        return status()

    # 1. Endpoint test nguyên bản
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

        try:
            validate_image(img_bytes)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        try:
            bbox_result = await engine.predict_bbox.remote.aio(img_bytes)
            masks = await segmentation_engine.predict_segmentation.remote.aio(img_bytes)
            return {**bbox_result, "masks": masks, "dino_embedding_preview": []}
        except Exception as exc:
            return JSONResponse(status_code=503, content={"error": str(exc)})

    # 2. QC Studio Object Detection (BBox) - YOLO26x + D-FINE-X
    @web_app.post("/bbox")
    async def bbox_infer(request: Request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("JSON body must be an object")
            if body.get("action") == "status":
                return status()
            img_b64 = body.get("image", "")
            thresh = float(body.get("threshold", 0.25))
            if not 0 <= thresh <= 1:
                raise ValueError("threshold must be between 0 and 1")
            if not isinstance(body.get("mapping", {}), dict):
                raise ValueError("mapping must be an object")
            img_bytes = decode_image(img_b64)
            res = await engine.predict_bbox.remote.aio(img_bytes, threshold=thresh)
            res["predictions"] = map_predictions(res["predictions"], body)
            for run in res.get("model_runs", []):
                run["supported_classes"] = [body.get("mapping", {}).get(n, n) for n in run.get("supported_classes", [])]
            if body.get("include_metadata"):
                return {"predictions": res.get("predictions", []), "model_runs": res.get("model_runs", [])}
            return res.get("predictions", [])
        except (ValueError, TypeError) as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except Exception as e:
            return JSONResponse(status_code=503, content={"error": str(e)})

    # 3. Endpoint chuẩn QC Studio cho Segmentation - YOLO26x-Seg + SAM 2.1 Large
    @web_app.post("/segmentation")
    @web_app.post("/sam2")
    async def seg_infer(request: Request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("JSON body must be an object")
            if body.get("action") == "status":
                return status()
            img_b64 = body.get("image", "")
            thresh = float(body.get("threshold", 0.25))
            if not 0 <= thresh <= 1:
                raise ValueError("threshold must be between 0 and 1")
            if not isinstance(body.get("mapping", {}), dict):
                raise ValueError("mapping must be an object")
            img_bytes = decode_image(img_b64)
            res = await segmentation_engine.predict_segmentation.remote.aio(img_bytes, threshold=thresh)
            if not isinstance(res, list):
                raise RuntimeError("Unexpected segmentation response")
            return map_predictions(res, body)
        except (ValueError, TypeError) as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except Exception as e:
            return JSONResponse(status_code=503, content={"error": str(e)})

    return web_app
