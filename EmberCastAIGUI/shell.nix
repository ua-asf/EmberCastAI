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
    libcanberra-gtk3
    xdotool
    dioxus-cli
  ];

  # For crates that use native OpenSSL bindings
  OPENSSL_DEV = "${pkgs.openssl.dev}";
  OPENSSL_DIR = "${pkgs.openssl.out}/lib";
  OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
  PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
  GTK_PATH="${pkgs.libcanberra-gtk3}/lib/gtk-3.0";
  XDG_DATA_DIRS="${pkgs.gtk3}/share:${pkgs.libcanberra}/share:$XDG_DATA_DIRS";

  shellHook = ''
    rustup default stable
    echo "Rust + OpenSSL development environment loaded."
  '';
}