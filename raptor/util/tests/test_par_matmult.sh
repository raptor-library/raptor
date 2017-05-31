#!/bin/bash

export TEST_PATH="../../../build/raptor/util/tests"

cd ../../../build
make
cd ../raptor/util/tests/
python test_par_matmult.py $TEST_PATH
mpirun -n 16 $TEST_PATH"/./test_par_matmult"

