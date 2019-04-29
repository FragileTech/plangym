FROM ubuntu:18.04

ENV BROWSER=/browser \
    LC_ALL=en_US.UTF-8

COPY requirements.txt plangym/requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends \
      ca-certificates locales libxml2 libxml2-dev gcc g++ wget make \
      python3 python3-dev python3-distutils libjpeg8-dev zlib1g-dev && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    wget -O - https://bootstrap.pypa.io/get-pip.py | python3 && \
    cd plangym && \
    pip3 install --no-cache-dir -r requirements.txt && \
    apt-get remove -y python3-dev libxml2-dev gcc g++ wget && \
    apt-get remove -y .*-doc .*-man >/dev/null && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo '#!/bin/bash\n\
\n\
echo\n\
echo "  $@"\n\
echo\n\' > /browser && \
    chmod +x /browser

COPY . plangym/
RUN cd plangym && pip3 install -e . --no-use-pep517
