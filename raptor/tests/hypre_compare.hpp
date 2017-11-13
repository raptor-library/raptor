#ifndef RAPTOR_TEST_HYPRE_COMPARE_HPP
#define RAPTOR_TEST_HYPRE_COMPARE_HPP
#include <assert.h>
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/external/hypre_wrapper.hpp"

void compare_dimensions(ParCSRMatrix* A, hypre_ParCSRMatrix* A_h,
        std::vector<int>& new_off_proc_map)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    hypre_CSRMatrix* A_h_diag = hypre_ParCSRMatrixDiag(A_h);
    hypre_CSRMatrix* A_h_offd = hypre_ParCSRMatrixOffd(A_h);
    HYPRE_Int diag_rows = hypre_CSRMatrixNumRows(A_h_diag);
    HYPRE_Int diag_cols = hypre_CSRMatrixNumCols(A_h_diag);
    HYPRE_Int offd_cols = hypre_CSRMatrixNumCols(A_h_offd);
    HYPRE_Int first_local_row = hypre_ParCSRMatrixFirstRowIndex(A_h);
    HYPRE_Int first_local_col = hypre_ParCSRMatrixFirstColDiag(A_h);
    HYPRE_Int global_rows = hypre_ParCSRMatrixGlobalNumRows(A_h);
    HYPRE_Int global_cols = hypre_ParCSRMatrixGlobalNumCols(A_h);

    int first_row, first_col;
    std::vector<int> proc_sizes(num_procs+1);
    std::vector<int> new_on_proc_col_map;

    // Gather raptor proc sizes, and create array of global col
    // indices (corresponding to hypre's)
    MPI_Allgather(&(A->local_num_rows), 1, MPI_INT, proc_sizes.data(),
            1, MPI_INT, MPI_COMM_WORLD);
    first_row = 0;
    for (int i = 0; i < rank; i++)
    {
        first_row += proc_sizes[i];
    }
    MPI_Allgather(&(A->on_proc_num_cols), 1, MPI_INT, proc_sizes.data(),
            1, MPI_INT, MPI_COMM_WORLD);
    first_col = 0;
    for (int i = 0; i < rank; i++)
    {
        first_col += proc_sizes[i];
    }

    // Form new off proc column map
    if (A->on_proc_num_cols)
    {
        new_on_proc_col_map.resize(A->on_proc_num_cols);
    }
    if (A->off_proc_num_cols)
    {
        new_off_proc_map.resize(A->off_proc_num_cols);
    }
    for (int i = 0; i < A->on_proc_num_cols; i++)
    {
        new_on_proc_col_map[i] = first_col + i;
    }
    if (!A->comm)
    {
        A->comm = new ParComm(A->partition, A->off_proc_column_map,
                A->on_proc_column_map);
    }
    A->comm->communicate(new_on_proc_col_map);
    std::copy(A->comm->recv_data->int_buffer.begin(),
            A->comm->recv_data->int_buffer.end(),
            new_off_proc_map.begin());

    // Make sure dimensions are correct
    assert(A->global_num_rows == global_rows);
    assert(A->global_num_cols == global_cols);
    assert(first_row == first_local_row);
    assert(first_col == first_local_col);
    assert(A->local_num_rows == diag_rows);
    assert(A->on_proc_num_cols == diag_cols);
}

void compare(ParCSRMatrix* A, hypre_ParCSRMatrix* A_h)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int first_row, first_col;
    int start, end;
    int ctrA, endA;

    std::vector<int> new_off_proc_map;
    compare_dimensions(A, A_h, new_off_proc_map);

    // Declare Hypre Variable/s
    hypre_CSRMatrix* A_h_diag = hypre_ParCSRMatrixDiag(A_h);
    hypre_CSRMatrix* A_h_offd = hypre_ParCSRMatrixOffd(A_h);
    HYPRE_Real* diag_data = hypre_CSRMatrixData(A_h_diag);
    HYPRE_Real* offd_data = NULL;
    HYPRE_Int offd_cols = hypre_CSRMatrixNumCols(A_h_offd);
    HYPRE_Int* diag_i = hypre_CSRMatrixI(A_h_diag);
    HYPRE_Int* diag_j = hypre_CSRMatrixJ(A_h_diag);
    HYPRE_Int* offd_i = hypre_CSRMatrixI(A_h_offd);
    HYPRE_Int* offd_j = hypre_CSRMatrixJ(A_h_offd);
    HYPRE_Int* col_map_offd = hypre_ParCSRMatrixColMapOffd(A_h);
    if (offd_cols)
    {
        offd_data = hypre_CSRMatrixData(A_h_offd);
    }

    A->sort();
    if (A->global_num_rows == A->global_num_cols)
    {
        A->on_proc->move_diag();
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        start = diag_i[i];
        end = diag_i[i+1];
        if (end - start)
        {
            if (diag_j[start] == i) start++;
            hypre_qsort1(diag_j, diag_data, start, end - 1);
        }

        start = offd_i[i];
        end = offd_i[i+1];
        if (end - start)
        {
            hypre_qsort1(offd_j, offd_data, start, end - 1);
        }
    }

    // Check that values are correct
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ctrA = A->on_proc->idx1[i];
        endA = A->on_proc->idx1[i+1];
        start = diag_i[i];
        end = diag_i[i+1];
        for (int j = start; j < end; j++)
        {
            if (ctrA < endA && A->on_proc->idx2[ctrA] == diag_j[j])
            {
                assert(fabs(A->on_proc->vals[ctrA] - diag_data[j]) < 1e-06);
                ctrA++;
            }
            else
            {
                assert(fabs(diag_data[j]) < 1e-15);
            }
        }

        ctrA = A->off_proc->idx1[i];
        endA = A->off_proc->idx1[i+1];
        start = offd_i[i];
        end = offd_i[i+1];
        for (int j = start; j < end; j++)
        {
            if (ctrA < endA && new_off_proc_map[A->off_proc->idx2[ctrA]]
                    == col_map_offd[offd_j[j]])
            {
                assert(fabs(A->off_proc->vals[ctrA] - offd_data[j]) < 1e-06);
                ctrA++;
            }
            else
            {
                assert(fabs(offd_data[j]) < 1e-15);
            }
        }
    }
}

void compareS(ParCSRMatrix* S, hypre_ParCSRMatrix* S_h)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int first_row, first_col;
    int start, end;
    int ctrA, endA;

    std::vector<int> new_off_proc_map;
    compare_dimensions(S, S_h, new_off_proc_map);

    // Declare Hypre Variables
    hypre_CSRMatrix* S_h_diag = hypre_ParCSRMatrixDiag(S_h);
    hypre_CSRMatrix* S_h_offd = hypre_ParCSRMatrixOffd(S_h);
    HYPRE_Real* diag_data = hypre_CSRMatrixData(S_h_diag);
    HYPRE_Real* offd_data = NULL;
    HYPRE_Int offd_cols = hypre_CSRMatrixNumCols(S_h_offd);
    HYPRE_Int* diag_i = hypre_CSRMatrixI(S_h_diag);
    HYPRE_Int* diag_j = hypre_CSRMatrixJ(S_h_diag);
    HYPRE_Int* offd_i = hypre_CSRMatrixI(S_h_offd);
    HYPRE_Int* offd_j = hypre_CSRMatrixJ(S_h_offd);
    HYPRE_Int* col_map_offd = hypre_ParCSRMatrixColMapOffd(S_h);
    if (offd_cols)
    {
        offd_data = hypre_CSRMatrixData(S_h_offd);
    }

    S->sort();
    if (S->global_num_rows == S->global_num_cols)
    {
        S->on_proc->move_diag();
    }
    for (int i = 0; i < S->local_num_rows; i++)
    {
        start = diag_i[i];
        end = diag_i[i+1];
        if (end - start)
        {
            hypre_qsort0(diag_j, start, end - 1);
        }

        start = offd_i[i];
        end = offd_i[i+1];
        if (end - start)
        {
            hypre_qsort0(offd_j, start, end - 1);
        }
    }

    for (int i = 0; i < S->local_num_rows; i++)
    {
        start = S->on_proc->idx1[i];
        end = S->on_proc->idx1[i+1];
        if (S->on_proc->idx2[start] == i)
        {
            start++;
        }
        ctrA = diag_i[i];
        assert(end - start == diag_i[i+1] - ctrA);
        for (int j = start; j < end; j++)
        {
            assert(diag_j[ctrA++] == S->on_proc->idx2[j]);
        }

        start = S->off_proc->idx1[i];
        end = S->off_proc->idx1[i+1];
        ctrA = offd_i[i];
        assert(end - start == offd_i[i+1] - ctrA);
        for (int j = start; j < end; j++)
        {
            assert(col_map_offd[offd_j[ctrA++]] == new_off_proc_map[S->off_proc->idx2[j]]);
        }
    }
}

#endif
