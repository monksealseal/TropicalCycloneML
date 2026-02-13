.PHONY: run setup docker cli help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies and create .env
	pip install -r requirements.txt
	@[ -f .env ] || cp .env.example .env
	@echo "\n  Edit .env and add your ANTHROPIC_API_KEY, then run: make run"

run: ## Start the API server (localhost:8000)
	uvicorn climate_agent.api.app:app --reload --port 8000

cli: ## Run the interactive CLI agent
	python -m climate_agent

docker: ## Build and run with Docker
	docker compose up --build

test-api: ## Quick smoke test of the API
	@echo "--- Register ---"
	@curl -s -X POST http://localhost:8000/api/v1/auth/register \
		-H "Content-Type: application/json" \
		-d '{"email":"test@example.com","password":"testtest123"}' | python -m json.tool
	@echo "\n--- Health ---"
	@curl -s http://localhost:8000/health | python -m json.tool
