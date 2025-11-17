# EmberCastAI

A frontend and backend application for predicting the growth of wildfires.

## Overview

The application consists of two main components:

- Backend: A FastAPI app that's build in a Docker container.
- Frontend: A Dioxus app that communicates with the backend and displays a before/after comparison of wildfire growth predictions.

## Usage

In order to run the model, you'll need to prep both the Docker container and the Dioxus app.

### Docker Container

You can very easily run the Docker container by pulling it from GHCR. To pull and run it, simply run the following command:

```bash
docker run --rm -p 8000:8000 ghcr.io/ua-asf/embercastai:latest
```

This will pull the latest version of the Docker container and run it, exposing port `8000` to your local machine. You can then access the FastAPI backend at `http://localhost:8000`.

From here, you can run the Dioxus app to interact with the backend at `localhost:8000`.

#### Alternative - Build Manually

This is highly discouraged, but if you want to build the Docker container manually.

I've provided the script `build_docker_image.sh`. This will build and host the Docker container for the FastAPI backend automatically. You will need to have Docker installed on your machine to run this script as well as `Nix`.

You will also need an OpenTopography API key to use the application. You can sign up for a free account at [OpenTopography](https://opentopography.org/). Once you have your API key, create a file in the root directory (`EmbercastAI`) name `env.sh` and add the following line to it:

```bash
export OPENTOPOGRAPHY_API_KEY="XXXXXXXXXXXXXXXXXXXXXX"
```

And replace `XXXXXXXXXXXXXXXXXXXXXX` with your actual API key.

Then, you can run:

```bash
bash build_docker_image.sh
```

You may be prompted for `sudo`, and after a while, you'll see something like:

```terminal
Loaded image: fastapi-app:latest
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

This means the backend is running and you can access it at `http://localhost:8000`.

### Dioxus App

The folder `EmberCastAIGUI` contains the Dioxus app. To run it, first install `Nix`. Then, run `nix-shell` in the `EmberCastAIGUI` directory to set up the environment. Finally, execute `dx serve`. From there, you can input your
