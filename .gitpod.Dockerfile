FROM python:3.10.12

USER gitpod
RUN sudo apt-get update && sudo apt-get install -y \
    python3-pip
