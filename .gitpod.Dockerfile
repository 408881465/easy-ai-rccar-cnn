FROM gitpod/workspace-full:latest

USER gitpod

# Install dependencies
RUN sudo apt-get update \
    && sudo apt-get install -y \
    python3.8 \
    python3-pip \
    && sudo rm -rf /var/lib/apt/lists/*

RUN python3 --version && pip3 --version

ENV PYTHONPATH="${PYTHONPATH}:/home/gitpod/.local/bin"
