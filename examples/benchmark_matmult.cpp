#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>

#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "core/types.hpp"
#include "gallery/matrix_IO.hpp"
#include "gallery/stencil.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/external/hypre_wrapper.hpp"

#include "util/linalg/spmv.hpp"
#include "util/linalg/matmult.hpp"

#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "_hypre_parcsr_mv.h"

//using namespace raptor;
int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    ParMatrix* A;
    ParMatrix* B;
    ParMatrix* C;

    // create data sets for the matrices
    int Ax = 3;
    int Ay = 3;
    int Annz = Ax * Ay;
    int Bx = 3;
    int By = 2;
    int Bnnz = Bx * By;

    data_t* v0 = new data_t[Annz];
    data_t* v1 = new data_t[Bnnz];

    for (int i = 0; i < Annz; i++)
    {
        v0[i] = ((double)rand())/RAND_MAX;
    }
    for (int i = 0; i < Bnnz; i++)
    {
        v1[i] = ((double)rand())/RAND_MAX;
    }  
  
    // create the matrices

    A = new ParMatrix(Ax,Ay,v0);
    B = new ParMatrix(Bx,By,v1);

    if (A->offd_num_cols)
    {
        A->offd->convert(CSC);
    }

    if (B->offd_num_cols)
    {
        B->offd->convert(CSC);
    }


    MPI_Datatype csr_type;
    create_csr_type(&csr_type);
    MPI_Type_commit(&csr_type);

    parallel_matmult(A, B, &C, csr_type);

    if (C->offd_num_cols)
       C->offd->convert(CSR);
    int row_start, row_end, global_row, global_col;
    double value;

    data_t Cdense[Ax][By] = {0};
    for (int i = 0; i < Ax; i++)
    {
        for (int j = 0; j < By; j++)
        {
            Cdense[i][j] = 0;
            for (int k = 0; k < Ay; k++)
            {
                Cdense[i][j] += v0[i*Ay + k] * v1[k*By + j];
            }
        }
    }

    for (int proc = 0; proc < num_procs; proc++)
    {
        if (proc == rank) for (int row = 0; row < C->local_rows; row++)
        {
            row_start = C->diag->indptr[row];
            row_end = C->diag->indptr[row+1];
            global_row = row + C->first_row;

            for (int j = row_start; j < row_end; j++)
            {
                global_col = C->diag->indices[j] + C->first_col_diag;
                value = C->diag->data[j];
                assert(fabs(value - Cdense[global_row][global_col]) < 1e-06);
            }

            if (C->offd_num_cols)
            {
                row_start = C->offd->indptr[row];
                row_end = C->offd->indptr[row+1];
                for (int j = row_start; j < row_end; j++)
                {
                    global_col = C->local_to_global[C->offd->indices[j]];
                    value = C->offd->data[j];
                    assert(fabs(value - Cdense[global_row][global_col]) < 1e-06);
                }
            }    
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }


    delete A;
    delete B;
    delete C;

    MPI_Type_free(&csr_type);    
    

    MPI_Finalize();

    return 0;
}
