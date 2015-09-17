# raptor

General, high performance algebraic multigrid solver

# build instructions

1. Create a build directory

```bash
mkdir build
```

2. Configure the build

```bash
cd build
ccmake ..
```
* press c to configure
* select/modify default options
  * ```ENABLE_UNIT_TESTS```: build unit tests and add to ctest
  * ```TEST_COMM_SIZE```: default MPI Communicator size for tests
* press g to generate and exit.
3. Build project

```bash
make
```
Note: make VERBOSE=1 if you want to see what flags are being used.

# test instructions

The build system is currently setup to use google test
with ctest. The build system currently looks through the source tree
and adds anything it finds in a test directory to ctest. For a simple
example, look at raptor/core/tests/ParVector.cpp.
