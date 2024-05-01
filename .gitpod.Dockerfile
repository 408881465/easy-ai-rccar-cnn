FROM python:3.8.10

USER gitpod
RUN sudo apt-get update && sudo apt-get install -y \
    python3-pip
