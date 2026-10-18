import base64
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from fastapi.testclient import TestClient

pytest.importorskip('modal')
spec = importlib.util.spec_from_file_location('vision_deploy_test', Path(__file__).parents[1] / 'deploy_vision.py')
vision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vision)


@pytest.fixture
def api(monkeypatch):
    calls = []
    state = {'fail': False, 'bad_seg': False}
    async def bbox(*args, **kwargs):
        calls.append('bbox')
        if state['fail']:
            raise RuntimeError('model unavailable')
        return {'predictions': [{'label': 'car', 'type': 'rectangle', 'points': [0, 0, 2, 2]}],
                'model_runs': [{'role': 'detector_a', 'status': 'success', 'supported_classes': ['car']}]}
    async def seg(*args, **kwargs):
        calls.append('seg')
        if state['fail']:
            raise RuntimeError('model unavailable')
        return {} if state['bad_seg'] else [{'label': 'road', 'type': 'mask', 'mask': [1, 0, 0, 0, 0]}]
    remote = lambda fn: SimpleNamespace(remote=SimpleNamespace(aio=fn))
    monkeypatch.setattr(vision, 'VisionEngine', lambda **kwargs: SimpleNamespace(predict_bbox=remote(bbox), predict_segmentation=remote(seg), predict_full=remote(bbox)))
    buf = io.BytesIO()
    Image.new('RGB', (4, 4)).save(buf, 'PNG')
    payload = {'image': base64.b64encode(buf.getvalue()).decode()}
    with TestClient(vision.fastapi_app.local()) as client:
        yield client, payload, calls, state


@pytest.mark.parametrize('route', ['/bbox', '/segmentation', '/sam2'])
@pytest.mark.parametrize('bad', [None, [], 42, {'image': 'broken'}, {'image': 'eA=='}, {'threshold': 2}, {'mapping': []}])
def test_invalid_requests_do_not_run_gpu(api, route, bad):
    client, payload, calls, _ = api
    body = {**payload, **bad} if isinstance(bad, dict) else bad
    response = client.post(route, content=__import__('json').dumps(body), headers={'Content-Type': 'application/json'})
    assert response.status_code == 400
    assert not calls


def test_route_isolation_mapping_and_metadata(api):
    client, payload, calls, _ = api
    response = client.post('/bbox', json={**payload, 'include_metadata': True, 'mapping': {'car': 'vehicle'}})
    assert response.status_code == 200
    assert response.json()['predictions'][0]['label'] == 'vehicle'
    assert response.json()['model_runs'][0]['supported_classes'] == ['vehicle']
    assert calls == ['bbox']
    response = client.post('/segmentation', json={**payload, 'mapping': {'road': 'surface'}})
    assert response.json()[0]['label'] == 'surface'
    assert calls == ['bbox', 'seg']


@pytest.mark.parametrize('route', ['/bbox', '/segmentation', '/sam2'])
def test_inference_failure_is_not_empty_success(api, route):
    client, payload, _, state = api
    state['fail'] = True
    assert client.post(route, json=payload).status_code == 503


def test_invalid_segmentation_contract_fails(api):
    client, payload, _, state = api
    state['bad_seg'] = True
    assert client.post('/segmentation', json=payload).status_code == 503


def test_status_does_not_claim_loaded_models(api):
    client, _, calls, _ = api
    assert client.get('/status').json()['model_readiness'] == 'not_checked'
    assert not calls


def test_invalid_upload_is_rejected(api):
    client, _, calls, _ = api
    assert client.post('/analyze', files={'file': ('bad.png', b'not an image', 'image/png')}).status_code == 400
    assert not calls

def test_semantic_coverage_keeps_rider_and_small_traffic_signs():
    torch = pytest.importorskip('torch')
    cls = vision.VisionEngine._get_user_cls()
    logits = torch.full((1, 19, 1, 19), -10.0)
    for index in range(19):
        logits[0, index, 0, index] = 10
    engine = SimpleNamespace(
        device='cpu', load_segmentation_models=lambda: None,
        seg_model=lambda **kwargs: SimpleNamespace(logits=logits),
        seg_processor=lambda **kwargs: SimpleNamespace(to=lambda device: {}),
        yolo_seg=lambda *args, **kwargs: [],
    )
    buf = io.BytesIO()
    Image.new('RGB', (19, 1)).save(buf, 'PNG')
    results = cls.predict_segmentation._get_raw_f()(engine, buf.getvalue())
    assert len(results) == 19
    assert {'rider', 'traffic sign', 'pedestrian'} <= {item['label'] for item in results}
    assert all(len(item['mask']) == 5 for item in results)


def test_segmentation_model_error_propagates():
    torch = pytest.importorskip('torch')
    cls = vision.VisionEngine._get_user_cls()
    def fail(**kwargs):
        raise RuntimeError('GPU failure')
    engine = SimpleNamespace(device='cpu', load_segmentation_models=lambda: None,
                             seg_model=fail, seg_processor=lambda **kwargs: SimpleNamespace(to=lambda device: {}))
    buf = io.BytesIO()
    Image.new('RGB', (2, 2)).save(buf, 'PNG')
    with pytest.raises(RuntimeError, match='Semantic segmentation inference failed'):
        cls.predict_segmentation._get_raw_f()(engine, buf.getvalue())


def test_background_is_not_falsely_judged_by_sam(monkeypatch):
    import qc_pipeline
    def encode(img):
        buf = io.BytesIO()
        img.save(buf, 'PNG')
        return base64.b64encode(buf.getvalue()).decode()
    gt = Image.new('L', (4, 4), 0)
    gt.putpixel((3, 3), 13)
    road = {'label': 'road', 'mask': [1] * 15 + [0] + [0, 0, 3, 3], 'model': 'segformer-b0'}
    car = {'label': 'car', 'mask': [1, 3, 3, 3, 3]}
    monkeypatch.setattr(qc_pipeline, 'infer_segmentation', lambda *a, **k: (
        [road, {**car, 'model': 'yolo26x-seg.pt'}, {**car, 'model': 'sam2.1_l.pt'}], {'name': 'mock'}))
    result = qc_pipeline.run_segmentation_pipeline(
        [{'name': 'a.png', 'mask_index': 'mask.png'}], {'a.png': encode(Image.new('RGB', (4, 4)))},
        {'mask.png': encode(gt)}, {'class_names': ['road', 'car'], 'mask_id_to_class': {'0': 'road', '13': 'car'}, 'use_gemini': False}, None)
    assert result['sorted_frames'][0]['error_count'] == 0

def test_bbox_missing_secondary_model_reports_failure_without_segment_load():
    pytest.importorskip('torch')
    cls = vision.VisionEngine._get_user_cls()
    class Yolo:
        names = {0: 'car'}
        def __call__(self, *args, **kwargs):
            return []
    engine = SimpleNamespace(device='cpu', load_bbox_models=lambda: None, yolo=Yolo(), dfine=None)
    buf = io.BytesIO()
    Image.new('RGB', (2, 2)).save(buf, 'PNG')
    result = cls.predict_bbox._get_raw_f()(engine, buf.getvalue())
    assert result['predictions'] == []
    assert [(r['role'], r['status']) for r in result['model_runs']] == [('detector_a', 'success'), ('detector_b', 'failed')]


def test_sam_batches_preserve_all_labels_and_boxes():
    torch = pytest.importorskip('torch')
    cls = vision.VisionEngine._get_user_cls()
    polygon = [[0, 0], [1, 0], [1, 1]]
    boxes = [SimpleNamespace(cls=torch.tensor([0]), conf=torch.tensor([0.8]), xyxy=torch.tensor([[0, 0, 1, 1]])) for _ in range(17)]
    class Yolo:
        names = {0: 'car'}
        def __call__(self, *a, **k):
            return [SimpleNamespace(boxes=boxes, masks=SimpleNamespace(xy=[polygon] * 17))]
    calls = []
    def sam(*args, bboxes, **kwargs):
        calls.append(len(bboxes))
        return [SimpleNamespace(masks=SimpleNamespace(xy=[polygon] * len(bboxes)))]
    engine = SimpleNamespace(device='cpu', load_segmentation_models=lambda: None,
                             seg_model=None, yolo_seg=Yolo(), sam=sam)
    buf = io.BytesIO()
    Image.new('RGB', (2, 2)).save(buf, 'PNG')
    result = cls.predict_segmentation._get_raw_f()(engine, buf.getvalue())
    assert calls == [16, 1]
    refined = [r for r in result if r['model'] == 'sam2.1_l.pt']
    assert len(refined) == 17
    assert all(r['label'] == 'car' and r['box'] == [0, 0, 1, 1] for r in refined)
