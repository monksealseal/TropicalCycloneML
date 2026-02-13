"""Run the OncologyML API server.

Usage:
    python -m cancer_agent.service
    # or
    uvicorn cancer_agent.service.app:create_app --factory --reload
"""

import uvicorn

from cancer_agent.service.app import create_app

if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
