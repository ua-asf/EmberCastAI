FROM nixos/nix

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . .

# Cache the nix dependencies
RUN nix-shell --run "echo Hello!"

# Set the entry point
ENTRYPOINT ["nix-shell", "--run", "dx serve"]