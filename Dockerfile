# Use an official GDAL image as the base image
FROM ghcr.io/osgeo/gdal:ubuntu-small-latest

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential

# Set the working directory in the container
WORKDIR /EmberCastAI

COPY . .

# Install the necessary dependencies
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

ENV PATH="/opt/venv/bin:$PATH"

CMD ["python", "get_and_crop_sar.py"]