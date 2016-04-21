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

#include <assert.h>

//using namespace raptor;
void assert_equals(data_t* v0, data_t* v1, int len, int first_v0)
{
    for (int i = 0; i < len; i++)
    {
        assert(fabs(v0[first_v0 + i] - v1[i]) < zero_tol);
    }
}

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    ParMatrix* A;
    ParVector* x;
    ParVector* b;

    // create data sets for the matrices
    data_t a_data[15] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    A = new ParMatrix(3, 5, a_data);
    b = new ParVector(A->global_rows, A->local_rows, A->first_row);
    x = new ParVector(A->global_cols, A->local_cols, A->first_col_diag);

    data_t x_set[5] = {2, 4, 6, 8, 10};
    data_t* x_data = NULL;
   
    if (x->local_n)
       x_data = x->local->data();

    for (int i = 0; i < x->local_n; i++)
    {
        x_data[i] = x_set[A->first_col_diag + i];
    }
    
    data_t correct_b[3];

    for (int i = 0; i < 3; i++)
    {
        correct_b[i] = a_data[i*5] * x_set[0];
        correct_b[i] += a_data[i*5 + 1] * x_set[1];
        correct_b[i] += a_data[i*5 + 2] * x_set[2];
        correct_b[i] += a_data[i*5 + 3] * x_set[3];
        correct_b[i] += a_data[i*5 + 4] * x_set[4];
    }

    data_t* b_data = NULL;
    if (b->local_n)
       b_data = b->local->data();

    data_t correct_x[5];
    for (int i = 0; i < 5; i++)
    {
        correct_x[i] = a_data[i] * correct_b[0];
        correct_x[i] += a_data[i + 5] * correct_b[1];
        correct_x[i] += a_data[i + 10] * correct_b[2];
    }

    parallel_spmv(A, x ,b, 1.0, 0.0, 0);
    assert_equals(correct_b, b_data, A->local_rows, A->first_row);

    parallel_spmv(A, x, b, 1.0, 0.0, 1);
    assert_equals(correct_b, b_data, A->local_rows, A->first_row);

    parallel_spmv_T(A, b, x, 1.0, 0.0);
    assert_equals(correct_x, x_data, A->local_cols, A->first_col_diag);

    MPI_Finalize();

    return 0;
}
