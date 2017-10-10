#include <assert.h>

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_matrix_IO.hpp"

using namespace raptor;

void compare(ParCSRMatrix* S, ParCSRBoolMatrix* S_rap)
{
    int start, end;

    S->sort();
    S->on_proc->move_diag();
    S_rap->sort();
    S_rap->on_proc->move_diag();

    assert(S->global_num_rows == S_rap->global_num_rows);
    assert(S->local_num_rows == S_rap->local_num_rows);
    assert(S->global_num_cols == S_rap->global_num_cols);
    assert(S->on_proc_num_cols == S_rap->on_proc_num_cols);
    assert(S->local_nnz == S_rap->local_nnz);

    assert(S->on_proc->idx1[0] == S_rap->on_proc->idx1[0]);
    assert(S->off_proc->idx1[0] == S_rap->off_proc->idx1[0]);
    for (int i = 0; i < S->local_num_rows; i++)
    {
        assert(S->on_proc->idx1[i+1] == S_rap->on_proc->idx1[i+1]);
        start = S->on_proc->idx1[i];
        end = S->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            assert(S->on_proc->idx2[j] == S_rap->on_proc->idx2[j]);
            //assert(fabs(S->on_proc->vals[j] - S_rap->on_proc->vals[j]) < 1e-06);
        }

        assert(S->off_proc->idx1[i+1] == S_rap->off_proc->idx1[i+1]);
        start = S->off_proc->idx1[i];
        end = S->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            assert(S->off_proc_column_map[S->off_proc->idx2[j]] == 
                    S_rap->off_proc_column_map[S_rap->off_proc->idx2[j]]);
            //assert(fabs(S->off_proc->vals[j] - S_rap->off_proc->vals[j]) < 1e-06);
        }
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    ParCSRMatrix* A;
    ParCSRMatrix* S;
    ParCSRBoolMatrix* S_rap;

    A = readParMatrix("rss_laplace_A0.mtx", MPI_COMM_WORLD, 1, 1);
    S = readParMatrix("rss_laplace_S0.mtx", MPI_COMM_WORLD, 1, 1);
    S_rap = A->strength(0.25);
    compare(S, S_rap);
    delete A;
    delete S;
    delete S_rap;

    A = readParMatrix("rss_laplace_A1.mtx", MPI_COMM_WORLD, 1, 0);
    S = readParMatrix("rss_laplace_S1.mtx", MPI_COMM_WORLD, 1, 0);
    S_rap = A->strength(0.25);
    compare(S, S_rap);
    delete A;
    delete S;
    delete S_rap;

    A = readParMatrix("rss_aniso_A0.mtx", MPI_COMM_WORLD, 1, 1);
    S = readParMatrix("rss_aniso_S0.mtx", MPI_COMM_WORLD, 1, 1);
    S_rap = A->strength();
    compare(S, S_rap);
    delete A;
    delete S;
    delete S_rap;

    A = readParMatrix("rss_aniso_A1.mtx", MPI_COMM_WORLD, 1, 0);
    S = readParMatrix("rss_aniso_S1.mtx", MPI_COMM_WORLD, 1, 0);
    S_rap = A->strength();
    compare(S, S_rap);
    delete A;
    delete S;
    delete S_rap;

    MPI_Finalize();
}
