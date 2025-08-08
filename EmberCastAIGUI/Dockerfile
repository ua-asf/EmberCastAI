FROM ubuntu:22.04

# Install Nix
RUN apt-get update && apt-get install -y curl && \
    apt-get install -y build-essential git

RUN sh <(curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install) --daemon

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . .

# Set the entry point
ENTRYPOINT ["nix-shell", "--run", "dx serve"]