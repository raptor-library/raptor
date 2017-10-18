#include <assert.h>

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "ruge_stuben/par_cf_splitting.hpp"
#include "ruge_stuben/par_interpolation.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_stencil.hpp"
#include <iostream>
#include <fstream>

using namespace raptor;

void compare(ParCSRMatrix* P, ParCSRMatrix* P_rap)
{
    int start, end;

    assert(P->global_num_rows == P_rap->global_num_rows);
    assert(P->global_num_cols == P_rap->global_num_cols);
    assert(P->local_num_rows == P_rap->local_num_rows);
    assert(P->on_proc_num_cols == P_rap->on_proc_num_cols);
    assert(P->off_proc_num_cols == P_rap->off_proc_num_cols);

    P->on_proc->sort();
    P->on_proc->move_diag();
    P->off_proc->sort();
    P_rap->on_proc->sort();
    P_rap->on_proc->move_diag();
    P_rap->off_proc->sort();

    assert(P->on_proc->idx1[0] == P_rap->on_proc->idx1[0]);
    assert(P->off_proc->idx1[0] == P_rap->off_proc->idx1[0]);
    for (int i = 0; i < P->local_num_rows; i++)
    {
        assert(P->on_proc->idx1[i+1] == P_rap->on_proc->idx1[i+1]);
        start = P->on_proc->idx1[i];
        end = P->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            assert(P->on_proc->idx2[j] == P_rap->on_proc->idx2[j]);
            assert(fabs(P->on_proc->vals[j] - P_rap->on_proc->vals[j]) < 1e-06);
        }

        assert(P->off_proc->idx1[i+1] == P_rap->off_proc->idx1[i+1]);
        start = P->off_proc->idx1[i];
        end = P->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            assert(P->off_proc->idx2[j] == P_rap->off_proc->idx2[j]);
            assert(fabs(P->off_proc->vals[j] - P_rap->off_proc->vals[j]) < 1e-06);
        }
    }
}

ParCSRMatrix* form_Prap(ParCSRMatrix* A, ParCSRBoolMatrix* S, char* filename, int* first_row_ptr, int* first_col_ptr)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int first_row, first_col;
    FILE* f;
    ParCSRMatrix* P_rap;
    std::vector<int> proc_sizes(num_procs);
    std::vector<int> splitting;
    if (A->local_num_rows)
    {
        splitting.resize(A->local_num_rows);
    }
    MPI_Allgather(&A->local_num_rows, 1, MPI_INT, proc_sizes.data(), 1, MPI_INT,
            MPI_COMM_WORLD);
    first_row = 0;
    for (int i = 0; i < rank; i++)
    {
        first_row += proc_sizes[i];
    }
    f = fopen(filename, "r");
    int cf;
    for (int i = 0; i < first_row; i++)
    {
        fscanf(f, "%d\n", &cf);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);

    // Get off proc states
    A->comm->communicate(splitting.data());
    P_rap = direct_interpolation(A, S, splitting, A->comm->recv_data->int_buffer);
    MPI_Allgather(&P_rap->on_proc_num_cols, 1, MPI_INT, proc_sizes.data(), 1, 
                MPI_INT, MPI_COMM_WORLD);
    first_col = 0;
    for (int i = 0; i < rank; i++)
    {
        first_col += proc_sizes[i];
    }

    *first_row_ptr = first_row;
    *first_col_ptr = first_col;

    return P_rap;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int first_row, first_col, col;
    int start, end;
    FILE* f;
    ParCSRMatrix* A;
    ParCSRBoolMatrix* S;
    ParCSRMatrix* P;
    ParCSRMatrix* P_rap;

    A = readParMatrix("../../tests/rss_laplace_A0.mtx", MPI_COMM_WORLD, 1, 1);
    S = A->strength(0.25);
    P_rap = form_Prap(A, S, "../../tests/rss_laplace_cf0.txt", &first_row, &first_col);
    P = readParMatrix("../../tests/rss_laplace_P0.mtx", MPI_COMM_WORLD, 1, 0, 
        P_rap->local_num_rows, P_rap->on_proc_num_cols, first_row, first_col);
    compare(P, P_rap);
    delete P_rap;
    delete P;
    delete S;
    delete A;

    A = readParMatrix("../../tests/rss_laplace_A1.mtx", MPI_COMM_WORLD, 1, 0);
    S = A->strength(0.25);
    P_rap = form_Prap(A, S, "../../tests/rss_laplace_cf1.txt", &first_row, &first_col);
    P = readParMatrix("../../tests/rss_laplace_P1.mtx", MPI_COMM_WORLD, 1, 0,
            P_rap->local_num_rows, P_rap->on_proc_num_cols, first_row, first_col);
    compare(P, P_rap);
    delete P;
    delete P_rap;
    delete S;
    delete A;

/*    
    A = readParMatrix("../../tests/rss_aniso_A0.mtx", MPI_COMM_WORLD, 1, 1);
    S = A->strength(0.0);
    P_rap = form_Prap(A, S, "../../tests/rss_aniso_cf0.txt", &first_row, &first_col);
    P = readParMatrix("../../tests/rss_aniso_P0.mtx", MPI_COMM_WORLD, 1, 0, 
        P_rap->local_num_rows, P_rap->on_proc_num_cols, first_row, first_col);
    compare(P, P_rap);
    delete P_rap;
    delete P;
    delete S;
    delete A;

    A = readParMatrix("../../tests/rss_aniso_A1.mtx", MPI_COMM_WORLD, 1, 0);
    S = A->strength(0.0);
    P_rap = form_Prap(A, S, "../../tests/rss_aniso_cf1.txt", &first_row, &first_col);
    P = readParMatrix("../../tests/rss_aniso_P1.mtx", MPI_COMM_WORLD, 1, 0,
            P_rap->local_num_rows, P_rap->on_proc_num_cols, first_row, first_col);
    compare(P, P_rap);
    delete P;
    delete P_rap;
    delete S;
    delete A;
*/

    MPI_Finalize();

    return 0;
}

