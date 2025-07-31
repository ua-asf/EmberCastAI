let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: [
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
    ]))
  ];
}
