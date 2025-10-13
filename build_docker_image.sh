#! bash
nix build #.docker
sudo docker load <result
sudo docker run -p 8000:8000 fastapi-app:latest
