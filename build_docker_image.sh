#! bash
nix build #.docker
sudo docker load <result
sudo docker run -v ./env.sh:/app/env.sh:ro -p 8000:8000 fastapi-app:latest
