[![Build Status](https://travis-ci.org/lukeolson/raptor.svg?branch=master)](https://travis-ci.org/lukeolson/raptor)

![](docs/logo/raptor-logo.png)

# raptor

RAPtor is a general, high performance algebraic multigrid solver.

# Requirements

- `MPI`
- `cmake`


# Build Instructions

1. Create a build directory
```bash
mkdir build
```
2. Configure the build

```bash
cd build
cmake [OPTIONS] ..
```

```bash
make
```
Note: make VERBOSE=1 if you want to see what flags are being used.

# Options

- `WITH_HYPRE`: 
    Includes hypre_wrapper in the build.  Hypre must be installed before
    building with this option.  If not installed to /usr/local, set the
    HYPRE_DIR option.
    
- `WITH_MFEM`:
    Includes mfem_wrapper, mfem files, and hypre_wrapper in the build. 
    Mfem, Metis, and Hypre must be installed before building with this
    option.  For any packages not installed to /usr/local, set the 
    directory option (<package>_DIR).
    
- `HYPRE_DIR`: 
    Sets the directory of hypre containing the include and lib folders

- `METIS_DIR`:
    Sets the directory of metis containing the include and libmetis folders

- `MFEM_DIR`:
    Sets the directory of mfem containing mfem.h and libmfem

# Unit Testing

The build system is currently setup to use google test with ctest. The build
system currently looks through the source tree and adds anything it finds in a
test directory to ctest. For a simple example, look at
raptor/core/tests/ParVector.cpp.
