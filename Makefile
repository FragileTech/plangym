current_dir = $(shell pwd)

PROJECT = plangym
n ?= 2
DOCKER_ORG = fragiletech
DOCKER_TAG ?= ${PROJECT}
ROM_FILE ?= "uncompressed_ROMs.zip"
ROM_PASSWORD ?= "NO_PASSWORD"
VERSION ?= latest
MUJOCO_PATH?=~/.mujoco

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
	python3 src/plangym/scripts/import_retro_roms.py

.PHONY: install-envs
install-envs:
	make -f Makefile.docker install-env-deps
	make install-mujoco

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

# ============ Development Commands ============

.PHONY: style
style:
	uv run ruff check --fix-only --unsafe-fixes tests src
	uv run ruff format tests src

.PHONY: check
check:
	uv run ruff check --diff tests src
	uv run ruff format --diff tests src

# ============ Test Commands ============

.PHONY: test
test: test-doctest test-parallel test-singlecore

.PHONY: test-parallel
test-parallel:
	MUJOCO_GL=egl PYVIRTUALDISPLAY_DISPLAYFD=0 SKIP_CLASSIC_CONTROL=1 \
	uv run pytest -n $n -s -o log_cli=true -o log_cli_level=info tests

.PHONY: test-singlecore
test-singlecore:
	PYTEST_XDIST_AUTO_NUM_WORKERS=1 PYVIRTUALDISPLAY_DISPLAYFD=0 \
	uv run pytest -s -o log_cli=true -o log_cli_level=info tests/control/test_classic_control.py

.PHONY: test-doctest
test-doctest:
	PYVIRTUALDISPLAY_DISPLAYFD=0 SKIP_CLASSIC_CONTROL=1 \
	uv run pytest --doctest-modules -n $n -s -o log_cli=true -o log_cli_level=info src

# ============ Code Coverage ============

.PHONY: codecov
codecov: codecov-singlecore codecov-parallel

.PHONY: codecov-parallel
codecov-parallel: codecov-parallel-1 codecov-parallel-2 codecov-parallel-3 codecov-vectorization

.PHONY: codecov-parallel-1
codecov-parallel-1:
	PYVIRTUALDISPLAY_DISPLAYFD=0 SKIP_CLASSIC_CONTROL=1 \
	uv run pytest -n $n -s -o log_cli=true -o log_cli_level=info \
	--cov=./ --cov-report=xml:coverage_parallel_1.xml --cov-config=pyproject.toml \
	tests/test_core.py tests/test_registry.py tests/test_utils.py

.PHONY: codecov-parallel-2
codecov-parallel-2:
	PYVIRTUALDISPLAY_DISPLAYFD=0 SKIP_CLASSIC_CONTROL=1 \
	uv run pytest -n $n -s -o log_cli=true -o log_cli_level=info \
	--cov=./ --cov-report=xml:coverage_parallel_2.xml --cov-config=pyproject.toml \
	tests/videogames

.PHONY: codecov-parallel-3
codecov-parallel-3:
	MUJOCO_GL=egl PYVIRTUALDISPLAY_DISPLAYFD=0 SKIP_CLASSIC_CONTROL=1 \
	uv run pytest -n $n -s -o log_cli=true -o log_cli_level=info \
	--cov=./ --cov-report=xml:coverage_parallel_3.xml --cov-config=pyproject.toml \
	tests/control

.PHONY: codecov-vectorization
codecov-vectorization:
	PYVIRTUALDISPLAY_DISPLAYFD=0 SKIP_CLASSIC_CONTROL=1 \
	uv run pytest -n 0 -s -o log_cli=true -o log_cli_level=info \
	--cov=./ --cov-report=xml:coverage_vectorization.xml --cov-config=pyproject.toml \
	tests/vectorization

.PHONY: codecov-singlecore
codecov-singlecore:
	PYTEST_XDIST_AUTO_NUM_WORKERS=1 PYVIRTUALDISPLAY_DISPLAYFD=0 \
	uv run pytest --doctest-modules -s -o log_cli=true -o log_cli_level=info \
	--cov=./ --cov-report=xml --cov-config=pyproject.toml \
	tests/control/test_classic_control.py

# ============ Documentation ============

.PHONY: docs
docs: build-docs serve-docs

.PHONY: build-docs
build-docs:
	uv run sphinx-build -b html docs/source docs/build

.PHONY: serve-docs
serve-docs:
	uv run python3 -m http.server --directory docs/build
