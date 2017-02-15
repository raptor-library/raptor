// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "random.hpp"

ParMatrix* random_mat(int global_rows, int global_cols, int nnz_per_row)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    ParMatrix* A;
    
    A = new ParMatrix(global_rows, global_cols);
    int local_nnz = nnz_per_row * A->local_num_rows;
    for (int i = 0; i < local_nnz; i++)
    {
        A->add_value(rand() % A->local_num_rows, rand() % global_cols, 1.0);
    }

    A->finalize();

    return A;

}
