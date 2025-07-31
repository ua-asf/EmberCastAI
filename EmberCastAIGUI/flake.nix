{
  description = "Python GUI app with embedded dependencies";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        python-env = pkgs.python3.withPackages (python-pkgs: [
          python-pkgs.pip
          python-pkgs.setuptools
          python-pkgs.pandas
          python-pkgs.requests
          python-pkgs.numpy
          python-pkgs.torch
          python-pkgs.scikit-learn
          python-pkgs.matplotlib
          python-pkgs.gdal
          python-pkgs.tqdm
          python-pkgs.python-dotenv
          python-pkgs.asf-search
          python-pkgs.tenacity
          python-pkgs.boto3
          python-pkgs.fiona
          python-pkgs.pyproj
          python-pkgs.scikit-image
        ]);

      in {
        packages.default = pkgs.writeShellApplication {
          name = "embercastaigui";
          runtimeInputs = [ python-env ];
          text = ''
            exec ${python-env}/bin/python3 ${./assets/scripts/process.py} "$@"
          '';
        };

        apps.default = flake-utils.lib.mkApp {
          drv = self.packages.${system}.default;
          exePath = "/bin/embercastaigui";
        };
      });
}