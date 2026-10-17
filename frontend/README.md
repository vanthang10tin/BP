# Frontend options

- `BlazorApp/` is the main UI on Render and in the Docker image. Its interactive server circuit connects through `/_blazor`; a successful HTML response alone does not prove the buttons work.
- `index.html`, `app.js`, `style.css` are the alternative static UI served when running FastAPI directly.
- The Windows launcher starts Blazor on `http://127.0.0.1:5000` and the API on `http://127.0.0.1:8000`. The Blazor workspace calls `/api/qc/run`; the static UI calls `/api/qc/check`.

Use `docker compose up --build` to check the production UI locally. Run `dotnet run --project tests/BlazorRegression -c Release` from the repository root for the upload/rendering regression checks.
