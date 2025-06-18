## Dockerfile 
## Instructions:
## 1. Build the image: docker build -t get-and-crop-sar .
## 2. Run the container: docker run -it --rm -v ${PWD}/organized_dataset:/EmberCastAI/organized_dataset get-and-crop-sar  

# Use latest GDAL ubuntu image
FROM ghcr.io/osgeo/gdal:ubuntu-small-latest

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential

# Set the working directory in the container
WORKDIR /EmberCastAI

# Copy the necessary files to the container
COPY . .

# Install the necessary Python libraries
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Get the path for the venv
ENV PATH="/opt/venv/bin:$PATH"

# Run the script
CMD ["python", "get_and_crop_sar.py"]