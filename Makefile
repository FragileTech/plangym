current_dir = $(shell pwd)

PROJECT = plangym
n ?= auto
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
