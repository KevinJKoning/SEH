# Use Ubuntu as the base image
FROM ubuntu:22.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12, pip, and other required packages
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python packages using uv
RUN /root/.cargo/bin/uv install jupyterlab notebook pandas numpy matplotlib seaborn

# Install Quarto
RUN wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.6.40/quarto-1.6.40-linux-amd64.deb \
    && apt-get update \
    && apt-get install -y ./quarto-1.6.40-linux-amd64.deb \
    && rm quarto-1.6.40-linux-amd64.deb \
    && rm -rf /var/lib/apt/lists/*