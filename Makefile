install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/

lint:
	flake8 .
	black . --check

format:
	black .
	isort .

run:
	python app.py

run-api:
	uvicorn api.main:app --reload

run-frontend:
	streamlit run frontend/app.py

docker-build:
	docker build -t multimodal-ai-assistant .

docker-run:
	docker run -p 8000:8000 multimodal-ai-assistant

clean:
	find . -type d -name __pycache__ -delete
	find . -name "*.pyc" -delete
