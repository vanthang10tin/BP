# Frontend options

- `index.html`, `app.js`, `style.css` are the static UI served by FastAPI on `/`. This is the UI used by the Render Blueprint and Docker image.
- `BlazorApp/` is an additional local interface. The Windows launcher starts it on `http://127.0.0.1:5000` and the API on `http://127.0.0.1:8000`. Its workspace flow calls `/api/qc/run`; the static UI currently calls `/api/qc/check`.

Run only the deploy-matching static UI locally from the repository root with `python backend/app.py`, then open `http://127.0.0.1:8000`.
