current_dir = $(shell pwd)

PROJECT = plangym

.PHONY: check
check:
	!(grep -R /tmp plangym/tests)
	flake8 --count
	pylint plangym
	black --check .

.PHONY: test
test:
	python3 -m pytest


.PHONY: docker-test
docker-test:
	find -name "*.pyc" -delete
	docker run --rm -it --network host -w /plangym --entrypoint python3 plangym -m pytest


.PHONY: docker-build
docker-build:
	docker build -t plangym .