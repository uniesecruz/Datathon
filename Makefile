## ----------------------------------------------------------------------
## Makefile is a shortcut for performing complex or repetitive commands.
## Instead of asking you to remember a bunch of commands, you can just run
## `make help` to see all the available commands.
## After that, select one of the commands below and run `make <command>`.
## ----------------------------------------------------------------------

YAML_FILE = bentofile.yaml

VERSION = $(shell python -c "import yaml; print(yaml.safe_load(open('$(YAML_FILE)'))['labels']['version'])")

help:  ## Display this message.
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

install: ## Install the dependencies.
	@echo "Installing dependencies..."
	pip install -r requirements/requirements.txt

install-dev: ## Install the dev dependencies.
	@echo "Installing dev dependencies..."
	pip install -r requirements/dev-requirements.txt

lint: ## Run ruff format
	ruff format .

fix: ## Run ruff fix
	ruff check . --fix

select-model: # Run experiments and select best model
	python -m src.model_selector

test: # Run tests
	pytest

clean:  ## Clean up cache and build files.
	@echo "Cleaning up build files..."
	python -c 'import shutil, os; [shutil.rmtree(d, ignore_errors=True) for d in [".pytest_cache", "build", "dist", ".mypy_cache"]];
for root, dirs, files in os.walk(".", topdown=False):
	for name in dirs:
		if name in ["__pycache__", ".eggs", ".egg-info"]:
			shutil.rmtree(os.path.join(root, name), ignore_errors=True)'

clean-mlflow:  ## Clean up mlflow and monitoring files
	@echo "Cleaning up build files..."
	rm -f mlruns monitoring

serve: # Serve bento model
	bentoml serve . --reload

list-models: # List models registered to bentoml
	bentoml models list

build: # Build bentoml
	bentoml build --version $(VERSION)

containerize: # Containerize builded bento
	bentoml containerize summarization:latest

load-test:
	locust -H http://127.0.0.1:3000