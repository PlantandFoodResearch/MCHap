FROM ubuntu:22.04

RUN apt-get update \
    && apt-get install -y \
    git \
    tabix \
    nano \
    python3-pip

RUN git clone  https://github.com/PlantandFoodResearch/MCHap.git   \
    && cd MCHap \
    && git checkout v0.9.0 \
    && pip install -r requirements.txt \
    && python3 setup.py sdist \
    && python3 -m pip install dist/mchap-*tar.gz

RUN pytest -vv MCHap

