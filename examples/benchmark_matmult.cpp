#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>

#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "gallery/matrix_IO.hpp"
#include "gallery/stencil.hpp"
#include "gallery/diffusion.hpp"

#include "util/linalg/spmv.hpp"
#include "util/linalg/matmult.hpp"

//using namespace raptor;
int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    ParMatrix* matrices[2];
    ParMatrix* A;

    // create data sets for the matrices
    data_t v0[9] = {1,0,0,0,0,0,0,0,0};
    data_t v1[15] = {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
    // create the matrices
    matrices[0] = new ParMatrix(3,3,v0);
    matrices[1] = new ParMatrix(3,5,v1);

    parallel_matmult(matrices[0], matrices[1], &A);

    A->offd->convert(CSR);
    int row_start, row_end, global_row, global_col;
    double value;

    // print the results of the mat mult
    for (int proc = 0; proc < num_procs; proc++)
    {
        if (proc == rank) for (int row = 0; row < A->local_rows; row++)
        {
            row_start = A->diag->indptr[row];
            row_end = A->diag->indptr[row+1];
            global_row = row + A->first_row;

            for (int j = row_start; j < row_end; j++)
            {
                global_col = A->diag->indices[j] + A->first_col_diag;
                value = A->diag->data[j];
                printf("(%d, %d) = %2.3e\n", global_row, global_col, value);
            }

            if (A->offd_num_cols)
            {
                row_start = A->offd->indptr[row];
                row_end = A->offd->indptr[row+1];
                for (int j = row_start; j < row_end; j++)
                {
                    global_col = A->local_to_global[A->offd->indices[j]];
                    value = A->offd->data[j];
                    printf("(%d, %d) = %2.3e\n", global_row, global_col, value);
                }
            }    
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}
