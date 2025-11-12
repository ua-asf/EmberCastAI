{
  description = "Docker Container for the EmberCastAI FastAPI service.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        python-with-overrides = pkgs.python3.override {
          packageOverrides = final: prev: {
            # asf-search
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

            asf-tools = final.buildPythonPackage rec {
              pname = "asf_tools";
              version = "0.8.3";
              format = "wheel";

              src = pkgs.fetchPypi {
                inherit pname version;
                format = "wheel";
                dist = "py3";
                python = "py3";
                abi = "none";
                platform = "any";
                hash = "sha256-cTeWyH+bQKxXubj9oNyBFPFlecL5+mlRp0B8LQojZvA=";
              };

              doCheck = false;
            };

            bmipy = final.buildPythonPackage rec {
              pname = "bmipy";
              version = "2.0.1";
              format = "wheel";

              src = pkgs.fetchPypi {
                inherit pname version;
                format = "wheel";
                dist = "py3";
                python = "py3";
                abi = "none";
                platform = "any";
                hash = "sha256-5c68+LJjtV6KCUuEXr4MTBfhqMCdpDZvxZO3aewDqbk=";
              };
             doCheck = false;
            };

            bmi-topography = final.buildPythonPackage rec {
              pname = "bmi_topography";
              version = "0.9.0";
              format = "wheel";

              src = pkgs.fetchPypi {
                inherit pname version;
                format = "wheel";
                dist = "py3";
                python = "py3";
                abi = "none";
                platform = "any";
                hash = "sha256-aMO50zSeyc/u9bodCNwnvCedy4vpY2l7swoF71rxvbg=";
              };

              propagatedBuildInputs = with final; [
                numpy
                requests
                bmipy
                rioxarray
              ];

              doCheck = false;
            };
          };
        };

        python-env = python-with-overrides.withPackages (python-pkgs: [
          python-pkgs.bmi-topography
          python-pkgs.boto3
          python-pkgs.setuptools
          python-pkgs.pandas
          python-pkgs.requests
          python-pkgs.numpy
          python-pkgs.torch
          python-pkgs.gdal
          python-pkgs.python-dotenv
          python-pkgs.asf-search
          python-pkgs.asf-tools
          python-pkgs.geopandas
          python-pkgs.fiona
          python-pkgs.owslib
          python-pkgs.scikit-learn
          python-pkgs.scikit-image
          python-pkgs.fastapi
          python-pkgs.tenacity
          python-pkgs.uvicorn
          python-pkgs.pydantic
        ]);

        python-dev = python-with-overrides.withPackages (python-pkgs: [
          python-pkgs.tqdm
          python-pkgs.joblib
          python-pkgs.pyproj
          pkgs.python3Packages.black
          pkgs.python3Packages.isort
          pkgs.python3Packages.mypy
          pkgs.coreutils
          pkgs.bash
        ]);

        dockerImage = pkgs.dockerTools.buildImage {
          name = "ua-asf/embercastai";
          tag = "latest";
          
          copyToRoot = pkgs.buildEnv {
            name = "app-root";
            paths = [
              python-env
              (pkgs.runCommand "copy-python-files" {} ''
                # matplotlib is special and makes me
                # write more code than I want to
                mkdir -p $out/tmp/matplotlib
                chmod -R 777 $out/tmp
                mkdir -p $out/app
                
                # Copy all used python files
                cp ${./.}/api.py $out/app/
                cp ${./.}/cde_data.py $out/app/
                cp ${./.}/dem.py $out/app/
                cp ${./.}/model.py $out/app/
                cp ${./.}/ship.py $out/app/
                cp ${./.}/cmr.py $out/app/
                cp ${./.}/geo.py $out/app/
                cp ${./.}/gamma.py $out/app/
                cp ${./.}/utils.py $out/app/
                cp ${./.}/wkt_processing.py $out/app/

                mkdir -p $out/app/checkpoints
                cp ${./.}/checkpoints/best_model.pth $out/app/checkpoints/best_model.pth
              '')
            ];
          };
          
          config = {
              Cmd = [
              "${pkgs.bash}/bin/bash"
              "-c"
              "exec ${python-env}/bin/uvicorn api:app --host 0.0.0.0 --port 8000"
            ];
            WorkingDir = "/app";
            ExposedPorts = {
              "8000/tcp" = {};
            };

            Env = [
              "MPLCONFIGDIR=/tmp/matplotlib"
              "TMPDIR=/tmp"
            ];

            User = "0000:0000";
          };
        };

      in {
        devShells.default = pkgs.mkShell {
          packages = [ python-env python-dev ];
          shellHook = ''
            source ./env.sh
          '';
        };
        
        packages = {
          default = dockerImage;
          docker = dockerImage;
        };
      });
}
