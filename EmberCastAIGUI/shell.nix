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
    libGL
    libxkbcommon
    mesa
    vulkan-loader
    wayland
    egl-wayland
    libepoxy
    cairo
    gdk-pixbuf
    atk
    gtk3
    dioxus-cli
  ];

  # Correct OpenSSL environment variables
  OPENSSL_DIR = "${pkgs.openssl.out}";  # Root directory, not lib subdirectory
  OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
  OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
  PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig:${pkgs.pkg-config}/lib/pkgconfig";
  
  # GTK/GUI related
  GTK_PATH = "${pkgs.libcanberra-gtk3}/lib/gtk-3.0";
  XDG_DATA_DIRS = "${pkgs.gtk3}/share:${pkgs.libcanberra}/share:$XDG_DATA_DIRS";
  WEBKIT_DISABLE_COMPOSITING_MODE = 1;
  LIBGL_ALWAYS_SOFTWARE = 1;
  GDK_BACKEND = "x11";

  shellHook = ''
    rustup default stable
    echo "Rust + OpenSSL development environment loaded."
    
    # Verify OpenSSL paths
    echo "OPENSSL_DIR: $OPENSSL_DIR"
    echo "OPENSSL_LIB_DIR: $OPENSSL_LIB_DIR"
    echo "OPENSSL_INCLUDE_DIR: $OPENSSL_INCLUDE_DIR"
  '';
}
