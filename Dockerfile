FROM ghcr.io/osgeo/gdal:ubuntu-full-3.12.0

# Install Python and pip
RUN apt-get update && apt-get install -y \
  python3-pip \
  python3-dev \
  bash \
  && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV TMPDIR=/tmp

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages --ignore-installed numpy -r requirements.txt

# Copy application files
COPY *.py ./
COPY checkpoints/best_model.pth ./checkpoints/

EXPOSE 8000

CMD ["exec uvicorn api:app --host 0.0.0.0 --port 8000"]
