{
  description = "Rust development for Dioxus Web";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            rustc
            cargo
            cargo-binstall
            rustfmt
            clippy
            rust-analyzer
            lld
            openssl
            pkg-config
            webkitgtk_4_1
            libcanberra-gtk3
            xdotool
            libxkbcommon
            libepoxy
            cairo
            gdk-pixbuf
            atk
            gtk3
          ];

          OPENSSL_DIR = "${pkgs.openssl.out}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
          PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
          
          shellHook = ''
            export PATH="$HOME/.cargo/bin:$PATH"
            echo "Rust + OpenSSL development environment loaded."
            
            if ! rustup target list --installed | grep -q "wasm32-unknown-unknown"; then
              echo "Installing wasm32-unknown-unknown target..."
              rustup target add wasm32-unknown-unknown
            fi
            
            if ! command -v dx &> /dev/null; then
              echo "Installing dioxus-cli..."
              cargo binstall dioxus-cli --force --no-confirm
            fi
          '';
        };

        # Add this apps section
        apps.default = {
          type = "app";
          program = toString (pkgs.writeShellScript "dx-build" ''
            export PATH="$HOME/.cargo/bin:$PATH"
            ${pkgs.lib.getExe pkgs.cargo-binstall} dioxus-cli --force --no-confirm 2>/dev/null || true
            exec dx bundle --release
          '');
        };
      }
    );
}
