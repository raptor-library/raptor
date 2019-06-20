[![Build Status](https://travis-ci.org/raptor-library/raptor.svg?branch=master)](https://travis-ci.org/raptor-library/raptor)

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
make -j 4
```
Note: `make VERBOSE=1` if you want to see what flags are being used.

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

The build system uses GoogleTest.  The build searches
the source tree and adds any
test directory to `ctest`. For a simple example, see
`raptor/core/tests/ParVector.cpp`.

To test:

```bash
make test
```

# Citing

<pre>
@MISC{BiOl2017,
      author = {Bienz, Amanda and Olson, Luke N.},
      title = {{RAPtor}: parallel algebraic multigrid v0.1},
      year = {2017},
      url = {https://github.com/lukeolson/raptor},
      note = {Release 0.1}
      }
</pre>

# Full Example

From the `examples` directory:
```
mpirun -n 4 ./example
```

This example is maintained in `raptor/examples/example.cpp`

```cpp

// Copyright (c) 2015-2017, Raptor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

// Include raptor
#include "raptor.hpp"

// This is a basic use case.
int main(int argc, char *argv[])
{
    // set rank and number of processors
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Create parallel matrix and vectors
    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

    // Timers
    double time_setup, time_solve, time_base;

    // Problems size and type
    int dim = 2;
    int n = 100;

    aligned_vector<int> grid;
    grid.resize(dim, n);

    // Anisotropic diffusion
    coarsen_t coarsen_type = CLJP;
    interp_t interp_type = ModClassical;
    relax_t relax_type = SOR;
    double eps = 0.001;
    double theta = M_PI/8.0;
    double* stencil = NULL;
    stencil = diffusion_stencil_2d(eps, theta);
    A = par_stencil_grid(stencil, grid.data(), dim);
    delete[] stencil;

    x = ParVector(A->global_num_cols, A->on_proc_num_cols);
    b = ParVector(A->global_num_rows, A->local_num_rows);

    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);

    // AMG parameters
    double strong_threshold = 0.25;

    // Create a multilevel object
    ParMultilevel* ml;

    // Setup Raptor Hierarchy
    MPI_Barrier(MPI_COMM_WORLD);
    time_base = MPI_Wtime();
    ml = new ParRugeStubenSolver(strong_threshold, coarsen_type, interp_type, Classical, relax_type);
    ml->setup(A);
    time_setup = MPI_Wtime() - time_base;

    // Print out information on the AMG hierarchy
    int64_t lcl_nnz;
    int64_t nnz;

    if (rank == 0) std::cout << "Level\tNumRows\tNNZ" << std::endl;
    if (rank == 0) std::cout << "-----\t-------\t---" << std::endl;
    for (int64_t i = 0; i < ml->num_levels; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
        lcl_nnz = Al->local_nnz;
        MPI_Reduce(&lcl_nnz, &nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) std::cout << i << "\t" << Al->global_num_rows << "\t" << nnz << std::endl;
    }

    // Solve Raptor Hierarchy
    MPI_Barrier(MPI_COMM_WORLD);
    time_base = MPI_Wtime();
    ml->solve(x, b);
    time_solve = MPI_Wtime() - time_base;

    MPI_Reduce(&time_setup, &time_base, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Raptor AMG Setup Time: %e\n", time_base);
    MPI_Reduce(&time_solve, &time_base, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Raptor AMG Solve Time: %e\n", time_base);

    // Delete AMG hierarchy
    delete ml;
    delete A;
    MPI_Finalize();

    return 0;
}

```

# License

This code is distributed under BSD: http://opensource.org/licenses/BSD-2-Clause

Please see `LICENSE.txt` and `COPYRIGHT.txt` for more information.

