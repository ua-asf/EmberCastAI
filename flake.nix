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
          python-pkgs.scikit-learn
          python-pkgs.scikit-image
          python-pkgs.joblib
          python-pkgs.fastapi
          python-pkgs.uvicorn
          python-pkgs.pydantic
          pkgs.bash
        ]);

        dockerImage = pkgs.dockerTools.buildImage {
          name = "fastapi-app";
          tag = "latest";
          
          copyToRoot = pkgs.buildEnv {
            name = "app-root";
            paths = [
              python-env
              pkgs.coreutils
              (pkgs.runCommand "copy-python-files" {} ''
                mkdir -p $out/app
                # matplotlib is special and makes me
                # write more code than I want to
                mkdir -p $out/tmp/matplotlib
                chmod -R 777 $out/tmp
                cp ${./.}/*.py $out/app/
              '')
            ];
          };
          
          config = {
            Cmd = [
              "${python-env}/bin/uvicorn"
              "api:app"
              "--host"
              "0.0.0.0"
              "--port"
              "8000"
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
          packages = [ python-env ];
        };
        
        packages = {
          default = dockerImage;
          docker = dockerImage;
        };
      });
}
