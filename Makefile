current_dir = $(shell pwd)

PROJECT = plangym
n ?= auto
DOCKER_ORG = fragiletech
DOCKER_TAG ?= ${PROJECT}
ROM_FILE ?= "uncompressed ROMs.zip"
ROM_PASSWORD ?= false
VERSION ?= latest
MUJOCO_PATH?=~/.mujoco

.POSIX:
style:
	black .
	isort .

.POSIX:
check:
	!(grep -R /tmp tests)
	flakehell lint ${PROJECT}
	pylint ${PROJECT}
	black --check ${PROJECT}

.PHONY: install-mujoco
install-mujoco:
	mkdir ${MUJOCO_PATH}
	wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
	tar -xvzf mujoco210-linux-x86_64.tar.gz -C ${MUJOCO_PATH}
	rm mujoco210-linux-x86_64.tar.gz

.PHONY: import-roms
import-roms:
	unzip -o -P ${ROM_PASSWORD} ${ROM_FILE}
	python3 import_retro_roms.py

.PHONY: install-envs
install-envs:
	python3 -m pip install -U pip wheel
	make -f Makefile.docker install-env-deps
	make install-mujoco

.PHONY: test-parallel
test-parallel:
	find -name "*.pyc" -delete
	DISABLE_RAY=True pytest -n $n -s -o log_cli=true -o log_cli_level=info

.PHONY: test-ray
test-ray:
	find -name "*.pyc" -delete
	pytest tests/test_ray.py -n 1 -s -o log_cli=true -o log_cli_level=info

.PHONY: test
test:
	xvfb-run -s "-screen 0 1400x900x24" make test-parallel test-ray

.PHONY: test-codecov
test-codecov:
	find -name "*.pyc" -delete
	xvfb-run -s "-screen 0 1400x900x24" pytest -n 1 -s -o log_cli=true -o log_cli_level=info --cov=./ --cov-report=xml --cov-config=pyproject.toml

.PHONY: pipenv-install
pipenv-install:
	rm -rf *.egg-info && rm -rf build && rm -rf __pycache__
	rm -f Pipfile && rm -f Pipfile.lock
	pipenv install --dev -r requirements-test.txt
	pipenv install --pre --dev -r requirements-lint.txt
	pipenv install -r requirements.txt
	pipenv install -e .
	pipenv lock

.PHONY: pipenv-test
pipenv-test:
	find -name "*.pyc" -delete
	pipenv run pytest -s

.PHONY: docker-shell
docker-shell:
	docker run --rm --gpus all -v ${current_dir}:/${PROJECT} --network host -w /${PROJECT} -it ${DOCKER_ORG}/${PROJECT}:${VERSION} bash

.PHONY: docker-notebook
docker-notebook:
	docker run --rm --gpus all -v ${current_dir}:/${PROJECT} --network host -w /${PROJECT} -it ${DOCKER_ORG}/${PROJECT}:${VERSION}

.PHONY: docker-build
docker-build:
	docker build --pull -t ${DOCKER_ORG}/${PROJECT}:${VERSION} . --build-arg ROM_PASSWORD=${ROM_PASSWORD}

.PHONY: docker-test
docker-test:
	find -name "*.pyc" -delete
	docker run --rm --network host -w /${PROJECT} -e CI=True -e DISABLE_RAY=True --entrypoint python3 ${DOCKER_ORG}/${PROJECT}:${VERSION} -m pytest -n $n -s -o log_cli=true -o log_cli_level=info
	docker run --rm --network host -w /${PROJECT} -e CI=True -e DISABLE_RAY=False --entrypoint python3 ${DOCKER_ORG}/${PROJECT}:${VERSION} -m pytest tests/test_ray.py -n 1 -s -o log_cli=true -o log_cli_level=info

.PHONY: docker-push
docker-push:
	docker push ${DOCKER_ORG}/${DOCKER_TAG}:${VERSION}
	docker tag ${DOCKER_ORG}/${DOCKER_TAG}:${VERSION} ${DOCKER_ORG}/${DOCKER_TAG}:latest
	docker push ${DOCKER_ORG}/${DOCKER_TAG}:latest
