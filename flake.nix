{
  description = "RAPtor algebraic multigrid library";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (_: prev: { mpi = prev.mpich; }) ];
        };
      in {
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "raptor";
          version = "0.4";

          src = ./.;

          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];

          buildInputs = [
            pkgs.mpi
            pkgs.blas
            pkgs.lapack
            pkgs.hypre
          ];

          cmakeFlags = [
            "-DWITH_MPI=ON"
            "-DWITH_HYPRE=ON"
            "-DHYPRE_DIR=${pkgs.hypre}"
            "-DENABLE_UNIT_TESTS=OFF"
            "-DBUILD_EXAMPLES=OFF"
          ];
        };

        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.mpi
            pkgs.blas
            pkgs.lapack
            pkgs.hypre
            pkgs.gtest
          ];
        };
      });
}
