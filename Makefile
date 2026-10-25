.PHONY: install artifacts lint fmt test run docker-build up down

install:
	pip install -r requirements-dev.txt

# rebuild serving artifacts from training outputs (after retraining / dvc pull)
artifacts:
	python scripts/export_artifacts.py --root . --out artifacts --model hybrid

lint:
	flake8 .
	black --check -l 120 src/api tests/test_api.py scripts
	isort --check --profile black -l 120 src/api tests/test_api.py scripts

fmt:
	black -l 120 src/api tests/test_api.py scripts
	isort --profile black -l 120 src/api tests/test_api.py scripts

test:
	pytest --cov=src/api --cov-report=term-missing

run:
	uvicorn src.api.main:app --reload --port 8000

docker-build:
	docker build -t book-recommender:dev .

up:
	docker compose up -d --build

down:
	docker compose down
