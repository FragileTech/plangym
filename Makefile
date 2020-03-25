current_dir = $(shell pwd)

PROJECT = plangym
VERSION ?= latest

.PHONY: check
check:
	!(grep -R /tmp tests)
	flake8 plangym --count
	pylint plangym
	black --check plangym

.PHONY: test
test:
	pytest -s


.PHONY: docker-test
docker-test:
	find -name "*.pyc" -delete
	docker run --rm -it --network host -w /plangym --entrypoint python3 plangym:${VERSION} -m pytest


.PHONY: docker-build
docker-build:
	docker build -t plangym .


.PHONY: docker-push
docker-push:
	docker push fragiletech/plangym:${VERSION}