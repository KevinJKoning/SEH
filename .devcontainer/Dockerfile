# Use Alpine as the base image
FROM alpine:latest

# Install Python 3.12, pip, virtualenv, and curl
RUN apk update && apk add --no-cache python3=3.12.8 python3-dev py3-pip wget curl

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install common libraries for the Jupyter ecosystem in the Python environment using uv
RUN uv install jupyterlab notebook pandas numpy matplotlib seaborn

# Install Quarto from Posit
RUN wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.6.40/quarto-1.6.40-linux-amd64.deb \
    && dpkg -i quarto-1.6.40-linux-amd64.deb \
    && rm quarto-1.6.40-linux-amd64.deb