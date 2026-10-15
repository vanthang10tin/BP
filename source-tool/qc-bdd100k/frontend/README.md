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
cd frontend/BlazorApp
dotnet run --no-launch-profile --urls http://localhost:5000
```

Open <http://localhost:5000>. Requires the .NET 10 SDK. The old `frontend/index.html` dashboard remains for compatibility on the Python server at port 8000; new Blazor workflows are on port 5000.

## Included workflow

- Choose BBox or semantic Segmentation workspace mode.
- Upload BDD100K JSON labels, optional guideline text/Markdown, and optional matching images for in-browser preview.
- Run QC and inspect prioritized frames, uploaded label boxes, issues, and guideline citations.
- Export reviewer Markdown, issue CSV, and guideline Markdown.
- Push issues to CVAT after configuring the backend-only `CVAT_URL` and `CVAT_ACCESS_TOKEN` values. Enter the target job id and first-frame offset. Verify frame ordering before pushing.

The QC run uses this machine's local CVAT Nuclio detector for BBox and local DINOv3 masks for semantic segmentation. Findings are candidates, not automatic corrections. No cloud VLM is called. Use uploaded train-ID masks and the matching ontology for segmentation.
