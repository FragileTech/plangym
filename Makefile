current_dir = $(shell pwd)

PROJECT = plangym
n ?= auto
DOCKER_ORG = fragiletech
DOCKER_TAG ?= ${PROJECT}
ROM_FILE ?= "uncompressed ROMs.zip"
ROM_PASSWORD ?= "NO_PASSWORD"
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
ifeq (${ROM_PASSWORD}, "NO_PASSWORD")
	unzip -o ${ROM_FILE}
else
	unzip -o -P ${ROM_PASSWORD} ${ROM_FILE}
endif
	python3 import_retro_roms.py

.PHONY: install-envs
install-envs:
	make -f Makefile.docker install-env-deps
	make install-mujoco

.PHONY: test-parallel
test-parallel:
	find . -name "*.pyc" -delete
	DISABLE_RAY=True pytest --doctest-modules -n $n -s -o log_cli=true -o log_cli_level=info

.PHONY: test-ray
test-ray:
	find . -name "*.pyc" -delete
	pytest tests/vectorization/test_ray.py -n 1 -s -o log_cli=true -o log_cli_level=info

.PHONY: doctest
doctest:
	DISABLE_RAY=True xvfb-run -s "-screen 0 1400x900x24" pytest plangym --doctest-modules -n $n -s -o log_cli=true -o log_cli_level=info

.PHONY: test
test:
	find . -name "*.pyc" -delete
	PYVIRTUALDISPLAY_DISPLAYFD=0 SKIP_CLASSIC_CONTROL=1 xvfb-run -s "-screen 0 1400x900x24" pytest -n auto -s -o log_cli=true -o log_cli_level=info tests
	PYVIRTUALDISPLAY_DISPLAYFD=0 xvfb-run -s "-screen 0 1400x900x24" pytest -s -o log_cli=true -o log_cli_level=info tests/control/test_classic_control.py

.PHONY: run-codecov-test
run-codecov-test:
	find . -name "*.pyc" -delete
	DISABLE_RAY=True pytest --doctest-modules -n $n -s -o log_cli=true -o log_cli_level=info --cov=./ --cov-report=xml --cov-config=pyproject.toml
	pytest tests/vectorization/test_ray.py -n 1 -s -o log_cli=true -o log_cli_level=info --cov-append --cov=./ --cov-report=xml --cov-config=pyproject.toml

.PHONY: test-codecov
test-codecov:
	xvfb-run -s "-screen 0 1400x900x24" make run-codecov-test

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
	docker run --rm --network host -w /${PROJECT} -e MUJOCO_GL=egl -e SKIP_RENDER=True -e DISABLE_RAY=True --entrypoint python3 ${DOCKER_ORG}/${PROJECT}:${VERSION} -m pytest -n $n -s -o log_cli=true -o log_cli_level=info
	docker run --rm --network host -w /${PROJECT} -e MUJOCO_GL=egl -e SKIP_RENDER=True -e DISABLE_RAY=False --entrypoint python3 ${DOCKER_ORG}/${PROJECT}:${VERSION} -m pytest tests/vectorization/test_ray.py -s -o log_cli=true -o log_cli_level=info

.PHONY: docker-push
docker-push:
	docker push ${DOCKER_ORG}/${DOCKER_TAG}:${VERSION}
	docker tag ${DOCKER_ORG}/${DOCKER_TAG}:${VERSION} ${DOCKER_ORG}/${DOCKER_TAG}:latest
	docker push ${DOCKER_ORG}/${DOCKER_TAG}:latest
