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

        # Override the python package set to include custom asf-search
        python-with-overrides = pkgs.python3.override {
          packageOverrides = final: prev: {
            asf-search = final.buildPythonPackage rec {
              pname = "asf_search";
              version = "10.1.0";
              format = "wheel";

              src = pkgs.fetchPypi {
                inherit pname version;
                format = "wheel";
                dist = "py3";
                python = "py3";
                abi = "none";
                platform = "any";
                hash = "sha256-bk3OBpy3MT8G551mEbPnDw+/GUX8eRzRfctC4KrNiWA=";
              };

              propagatedBuildInputs = with final; [
                requests
                python-dateutil
                shapely
                pytz
                dateparser
              ];

              doCheck = false;
            };
          };
        };

        python-env = python-with-overrides.withPackages (python-pkgs: [
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
          python-pkgs.asf-search  # Now uses the overridden version
          python-pkgs.tenacity
          python-pkgs.boto3
          python-pkgs.fiona
          python-pkgs.pyproj
          python-pkgs.scikit-image
          python-pkgs.joblib

          python-pkgs.ipython
          python-pkgs.black
        ]);
      in {
        packages.default = pkgs.writeShellApplication {
          name = "embercastaigui";
          runtimeInputs = [ python-env ];
          text = ''
            exec ${python-env}/bin/python3 assets/scripts/process.py "$@"
          '';
        };

        apps.default = flake-utils.lib.mkApp {
          drv = self.packages.${system}.default;
          exePath = "/bin/embercastaigui";
        };

        devShells.default = pkgs.mkShell {
          packages = [ python-env ];
        };
      });
}
