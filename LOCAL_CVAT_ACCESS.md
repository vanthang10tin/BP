# Local CVAT access recovery

**Current local address:** `http://127.0.0.1:8080`

## What was wrong

The existing CVAT containers and their persistent volumes were running, but the CVAT UI and server had no published Windows host ports. The UI container listens on port 8000 internally and the CVAT API server listens on 8080. The frontend also needs a same-origin gateway to route `/api`, `/static`, and `/data` requests to the API server.

## Recovery applied

Started a small Nginx gateway container named `cvat-local-gateway` on the existing `cvat_cvat` Docker network. It binds only to `127.0.0.1:8080`, routes the CVAT UI to `cvat_ui:8000`, and routes API/static/data to `cvat_server:8080`. It uses the existing `cvat/ui:dev` image and the checked-in [gateway config](source-tool/qc-bdd100k/infra/local-cvat-gateway.conf).

The existing CVAT containers, database, annotation files, and Docker volumes were not recreated or modified. The gateway has `unless-stopped` restart behavior and mounts its config read-only.

## Verification

- `http://127.0.0.1:8080/` serves the CVAT login page (HTTP 200).
- `/api/server/about` and `/api/auth/rules` return HTTP 200 through the same origin.
- The CVAT UI bundle and the local projection overlay JavaScript return HTTP 200.
- `/d14-overlay/private/config.json` returns HTTP 401 when unauthenticated, as expected for the protected overlay config.

## Sending QC issues from QC Studio

The CVAT gateway alone does not authorize QC Studio. In QC Studio's Prepare screen, use the Connect to CVAT card to enter the CVAT URL and a scoped access token. QC Studio writes these values only to `source-tool/qc-bdd100k/backend/.env`; the saved token is not returned in the API response or persisted by the frontend. The backend applies the new configuration immediately, so a restart is not required.

For manual setup, configure the backend `.env`:

```text
CVAT_URL=http://127.0.0.1:8080
CVAT_ACCESS_TOKEN=<scoped-CVAT-token>
```

Keep the real token out of source control, browser code, and screenshots. In Export, select the CVAT Job and load its frame map before pushing. If using a frame offset instead, explicitly confirm the frame ordering. The backend verifies the job and frame range, creates issues only at supplied evidence boxes, and updates Stage/State only after issue creation has no failures. A partial or failed API response is reported as such; it is not presented as a successful return.

## Operations

Check gateway state:

```powershell
docker ps --filter name=cvat-local-gateway
docker logs --tail 50 cvat-local-gateway
```

Stop or remove only the gateway (this does not stop CVAT):

```powershell
docker stop cvat-local-gateway
docker rm cvat-local-gateway
```

If the BP folder is moved, update the bind-mounted config path by recreating the gateway with the new config path. The Compose source reported by Docker currently points to a host path with no compose YAML, so this gateway is the reversible local access repair until that source is restored.
