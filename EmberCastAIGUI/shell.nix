{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    rustc
    cargo
    rustfmt
    clippy
    rustup
    openssl
    pkg-config
    webkitgtk_4_1
    xdotool
    dioxus-cli
  ];

  # For crates that use native OpenSSL bindings
  OPENSSL_DEV = "${pkgs.openssl.dev}";
  OPENSSL_DIR = "${pkgs.openssl.out}/lib";
  OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
  PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";

  shellHook = ''
    rustup default stable
    echo "Rust + OpenSSL development environment loaded."
  '';
}