# CLAUDE.md - Project Instructions for Claude Code

## Web Dashboard Deployment

This project includes a public web dashboard at `web/` that auto-deploys via GitHub Pages.

### How it works
- All web files live in `web/` (pure HTML/CSS/JS, no build step)
- On push to `main`, GitHub Actions deploys `web/` to GitHub Pages automatically
- The live site URL is: `https://<github-username>.github.io/TropicalCycloneML/`

### Workflow for making web changes
1. Edit files in `web/` (index.html, style.css, app.js)
2. Commit and push to main
3. GitHub Actions deploys automatically (~60 seconds)
4. The site is live at the GitHub Pages URL

### One-time setup (repo owner)
1. Go to repo Settings > Pages
2. Under "Build and deployment", select **GitHub Actions** as the source
3. The workflow at `.github/workflows/deploy-web.yml` handles the rest

### Key files
- `web/index.html` - Main dashboard page
- `web/style.css` - Dark-themed styles
- `web/app.js` - Storm data fetching, map rendering, UI logic
- `.github/workflows/deploy-web.yml` - Auto-deployment workflow

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
