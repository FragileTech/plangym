FROM ubuntu:22.04


RUN apt-get update && \
	apt-get install -y --no-install-suggests --no-install-recommends \
    make cmake git xvfb build-essential curl ssh wget ca-certificates swig \
    libegl1-mesa-dev libglu1-mesa libgl1-mesa-glx \
    python3 python3-dev && \
    wget -O - https://bootstrap.pypa.io/get-pip.py | python3

RUN python3 -m pip install uv
COPY src ./src
COPY requirements.lock .
COPY pyproject.toml .
COPY src/plangym/scripts/import_retro_roms.py .
COPY LICENSE .
COPY README.md .
COPY ROMS.zip .

RUN uv pip install --no-cache --system -r requirements.lock
RUN python3 src/plangym/scripts/import_retro_roms.py