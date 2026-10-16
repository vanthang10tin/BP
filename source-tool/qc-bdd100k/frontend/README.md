# QC Studio frontend

The primary new interface is the Blazor app committed under `BlazorApp/`. It uses the local Python API at `http://localhost:8000` by default. Configure another API URL with `QcApi:BaseUrl` in Blazor configuration.

## Run

Start the backend in one terminal from `source-tool/qc-bdd100k`:

```powershell
python -m pip install -r requirements.txt
python backend/app.py
```

Start the Blazor frontend in a second terminal:

```powershell
cd source-tool/qc-bdd100k/frontend/BlazorApp
dotnet run --no-launch-profile --urls http://localhost:5000
```

Open <http://localhost:5000>. Requires the .NET 10 SDK. The old `frontend/index.html` dashboard remains for compatibility on the Python server at port 8000; new Blazor workflows are on port 5000.

## Included workflow

- Choose BBox or semantic Segmentation workspace mode.
- Upload frame-level BDD100K/COCO/YOLO-JSON/normalized annotations, optional guideline text/Markdown, and optional matching images for in-browser preview. CVAT task label definitions (for example, a JSON array of `{ name, id, color, type, attributes }`) are not frame annotations and cannot create a QC run.
- Run QC and inspect prioritized frames, uploaded label boxes, issues, and guideline citations.
- Export reviewer Markdown, issue CSV, and guideline Markdown.
- Configure `CVAT_URL` and `CVAT_ACCESS_TOKEN` in the Connect to CVAT setup card; they are saved only to the backend's `.env` file. Select a CVAT job and use **Go to Review** to open that job's CVAT review page in a new tab. Enter the target job id and first-frame offset before pushing QC issues, and verify frame ordering.

The QC run uses this machine's local CVAT Nuclio detector for BBox and local DINOv3 masks for semantic segmentation. Findings are candidates, not automatic corrections. No cloud VLM is called. Use uploaded train-ID masks and the matching ontology for segmentation.

Set-ExecutionPolicy -Scope Process Bypass -Force
>> .\run_qc_studio.ps1 and .\stop_qc_studio.ps1