FROM ubuntu:22.04

# Set timezone
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python
RUN apt-get update && \
    apt-get install -y \
                    python3-pip \
                    python-is-python3

# Install opencv reqs
RUN apt-get update && apt-get install -y \
                            libopencv-dev \
                            python3-opencv \
                            ffmpeg

RUN python -m pip install uv

WORKDIR /work/lxd-io
COPY requirements.txt .
RUN uv pip install -r requirements.txt --system
