# CLAUDE.md - Project Instructions for Claude Code

## Web Dashboard Deployment

This project includes a public web dashboard at `web/` with two deployment options:
**Google Cloud Run** (primary) and **GitHub Pages** (fallback).

### Key files
- `web/index.html` - Main dashboard page
- `web/style.css` - Dark-themed styles
- `web/app.js` - Storm data fetching, map rendering, UI logic
- `web/Dockerfile` - Container image for Cloud Run (nginx)
- `web/nginx.conf` - Nginx config with health check and gzip
- `.github/workflows/deploy-cloudrun.yml` - Auto-deploy to Cloud Run
- `.github/workflows/deploy-web.yml` - Auto-deploy to GitHub Pages
- `scripts/deploy-cloudrun.sh` - Manual one-command deploy to Cloud Run

### Workflow for making web changes
1. Edit files in `web/` (index.html, style.css, app.js)
2. Commit and push to main
3. GitHub Actions auto-deploys to Cloud Run (~90 seconds)
4. The site is live at your Cloud Run URL

### Google Cloud Run deployment (primary)

#### Manual deploy (one command)
```bash
./scripts/deploy-cloudrun.sh YOUR_PROJECT_ID us-central1
```

#### Auto-deploy via GitHub Actions
On push to `main`, `.github/workflows/deploy-cloudrun.yml` builds and deploys automatically.

#### One-time GCP setup
1. Create a GCP project and enable Cloud Run, Artifact Registry, and IAM APIs
2. Create an Artifact Registry Docker repo: `gcloud artifacts repositories create tropicalcycloneml --repository-format=docker --location=us-central1`
3. Set up Workload Identity Federation for GitHub Actions (see: https://github.com/google-github-actions/auth#workload-identity-federation)
4. Add these GitHub repo secrets:
   - `GCP_PROJECT_ID` - Your GCP project ID
   - `WIF_PROVIDER` - Workload Identity Provider resource name
   - `WIF_SERVICE_ACCOUNT` - Service account email

### GitHub Pages deployment (fallback)

#### One-time setup
1. Go to repo Settings > Pages
2. Under "Build and deployment", select **GitHub Actions** as the source
3. The workflow at `.github/workflows/deploy-web.yml` handles the rest
4. Live at: `https://<github-username>.github.io/TropicalCycloneML/`

### Adding new pages
Add new `.html` files to `web/`. They deploy automatically. No build step needed.

## Python Climate Agent

### Running
```bash
pip install -e .
climate-agent
```

### Testing
```bash
pytest
```

### Linting
```bash
ruff check .
```
