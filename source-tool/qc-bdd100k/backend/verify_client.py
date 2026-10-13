import urllib.request
import json
import sys

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

base = 'http://localhost:8000'

try:
    # 1. Test index page
    req = urllib.request.urlopen(f'{base}/')
    html = req.read().decode('utf-8')
    print(f"[1] GET / : {req.status} | HTML length: {len(html)} bytes")
    assert 'QC Studio 2D' in html, 'Title missing'
    assert 'Hàng Đợi Review' in html, 'Review queue missing'
    assert '⚡ 1-Click Auto-Heal' in html, 'Auto-heal button missing'
    assert '📋 Xuất Phiếu Trả Job' in html, 'Export feedback button missing'

    # 2. Test Guideline API GET
    req = urllib.request.urlopen(f'{base}/api/guideline')
    guide_data = json.loads(req.read().decode('utf-8'))
    articles = guide_data.get('guideline', {}).get('articles', {})
    print(f"[2] GET /api/guideline : {req.status} | Articles count: {len(articles)}")
    for k, v in list(articles.items())[:3]:
        print(f"    - [{k}] {v.get('title')}: {v.get('rule')[:60]}...")

    # 3. Test Sample QC API GET
    req = urllib.request.urlopen(f'{base}/api/qc/sample')
    sample_qc = json.loads(req.read().decode('utf-8'))
    summary = sample_qc.get('summary', {})
    sorted_frames = sample_qc.get('sorted_frames', [])
    print(f"[3] GET /api/qc/sample : {req.status}")
    print(f"    - Total frames: {summary.get('total_frames')}")
    print(f"    - Auto pass count: {summary.get('auto_pass_count')}")
    print(f"    - Workload reduction: {summary.get('workload_reduction_pct')}%")
    print(f"    - Critical count: {summary.get('critical_count')}")
    print(f"    - Sorted frames count: {len(sorted_frames)}")

    # Check sorting order: 1 (Critical) -> 2 (High) -> 3 (Medium) -> 4 (Pass)
    ranks = [item.get('priority_rank') for item in sorted_frames]
    print(f"    - Queue Priority Ranks: {ranks}")
    assert ranks == sorted(ranks), f'Queue is not sorted ascending by priority rank: {ranks}'

    for idx, f in enumerate(sorted_frames):
        print(f"    Frame {idx+1}: {f.get('frame_name')} | Rank: {f.get('priority_rank')} ({f.get('priority_level')}) | Rec: {f.get('recommendation')} | Models: {f.get('flagged_models')}")
        for err in f.get('errors', []):
            print(f"       -> Defect: {err.get('error_type')} | Reason: {err.get('explanation')} | Guideline: {err.get('guideline_rule')}")

    # 4. Test Auto-heal API POST
    raw_data = sample_qc.get('raw_data', [])
    req_heal = urllib.request.Request(f'{base}/api/auto-heal', data=json.dumps(raw_data).encode('utf-8'), headers={'Content-Type': 'application/json'})
    res_heal = urllib.request.urlopen(req_heal)
    heal_res = json.loads(res_heal.read().decode('utf-8'))
    print(f"[4] POST /api/auto-heal : {res_heal.status} | Stats: {heal_res.get('stats')}")
    assert heal_res.get('status') == 'success'
    assert 'healed_data' in heal_res

    # 5. Test Export Feedback API POST
    req_fb = urllib.request.Request(f'{base}/api/export/feedback', data=json.dumps({'sorted_frames': sorted_frames, 'annotator_name': 'NguyenVanA'}).encode('utf-8'), headers={'Content-Type': 'application/json'})
    res_fb = urllib.request.urlopen(req_fb)
    fb_res = json.loads(res_fb.read().decode('utf-8'))
    print(f"[5] POST /api/export/feedback : {res_fb.status}")
    print("    Preview Feedback Markdown:\n" + "\n".join("      " + line for line in fb_res.get('feedback_markdown', '').splitlines()[:6]))

    print("\n========================================================")
    print("===> ALL 5 API & UI CLIENT INTEGRATION CHECKS PASSED! <===")
    print("========================================================")

except Exception as e:
    print(f"FAILED: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
