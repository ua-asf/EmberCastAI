#! bash

if command -v "nix" >/dev/null 2>&1; then
  echo "Building with nix."
  nix build #.docker
  sudo docker load <result
else
  echo "Building with Docker."
  docker build . -t fastapi-app
fi

sudo docker run -v ./env.sh:/app/env.sh:ro -p 8000:8000 fastapi-app:latest
