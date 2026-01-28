.PHONY: install lint test clean

install:
	pip install -r requirements.txt

lint:
	flake8 module1_object_detection/ module2_point_cloud/ module3_map_alignment/ utils/
	black --check .
	isort --check .

format:
	black .
	isort .

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
