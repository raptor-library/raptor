// EXPECT_EQ and ASSERT_EQ are macros
// EXPECT_EQ test execution and continues even if there is a failure
// ASSERT_EQ test execution and aborts if there is a failure
// The ASSERT_* variants abort the program execution if an assertion fails 
// while EXPECT_* variants continue with the run.


#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "ruge_stuben/par_cf_splitting.hpp"
#include "ruge_stuben/par_interpolation.hpp"
#include "multilevel/multilevel.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/par_stencil.hpp"
#include <iostream>
#include <fstream>

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
    MPI_Finalize();

} // end of main() //

void read_splitting(int local_rows, char* filename, std::vector<int>& splitting)
{
    int num_procs;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int cf;
    int first_row = 0;
    int first_col = 0;
    int local_cols = 0;
    std::vector<int> proc_sizes(num_procs);

    if (local_rows)
    {
        splitting.resize(local_rows);
    }
    
    MPI_Allgather(&local_rows, 1, MPI_INT, proc_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < rank; i++)
    {
        first_row += proc_sizes[i];
    }

    FILE* f = fopen(filename, "r");
    for (int i = 0; i < first_row; i++)
    {
        fscanf(f, "%d\n", &cf);
    }
    for (int i = 0; i < local_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);
}


void compare_splitting(int local_rows, std::vector<int>& splitting, char* filename)
{
    int num_procs;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int cf;
    int first_row = 0;
    int first_col = 0;
    int local_cols = 0;
    std::vector<int> proc_sizes(num_procs);
    
    MPI_Allgather(&local_rows, 1, MPI_INT, proc_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < rank; i++)
    {
        first_row += proc_sizes[i];
    }

    FILE* f = fopen(filename, "r");
    for (int i = 0; i < first_row; i++)
    {
        fscanf(f, "%d\n", &cf);
    }
    for (int i = 0; i < local_rows; i++)
    {
        fscanf(f, "%d\n", &cf);
        //assert(splitting[i] == cf);
        if (splitting[i] != cf)
            printf("Rank %d Splitting[%d] = %d, cf %d\n", rank, i, splitting[i], cf);
    }
    fclose(f);
}

void find_col_dimensions(char* filename, int first_row, int local_rows, 
        int* first_col_ptr, int* local_cols_ptr)
{
    int cf;
    int first_col = 0;
    int local_cols = 0;
    
    FILE* f = fopen(filename, "r");
    for (int i = 0; i < first_row; i++)
    {
        fscanf(f, "%d\n", &cf);
        first_col += cf;
    }
    for (int i = 0; i < local_rows; i++)
    {
        fscanf(f, "%d\n", &cf);
        local_cols += cf;
    }
    fclose(f);

    *first_col_ptr = first_col;
    *local_cols_ptr = local_cols;
}

int condense(Matrix* A, std::vector<int>& A_idx2, std::vector<int>& A_orig_to_new)
{
    if (A->nnz == 0) return 0;

    std::vector<int> A_exists(A->n_cols, false);
    A_orig_to_new.resize(A->n_cols);
    A_idx2.resize(A->nnz);
    for (std::vector<int>::iterator it = A->idx2.begin();
            it != A->idx2.end(); ++it)
    {
        A_exists[*it] = true;
    }
    int ctr = 0;
    for (int i = 0; i < A->n_cols; i++)
    {
        if (A_exists[i])
        {
            A_orig_to_new[i] = ctr++;
        }
    }
    for (std::vector<int>::iterator it = A->idx2.begin();
            it != A->idx2.end(); ++it)
    {
        *it = A_orig_to_new[*it];
    }

    return ctr;
}

void compare(ParMatrix* P, ParCSRMatrix* P_rap)
{
    int start, end;

    ASSERT_EQ(P->global_num_rows, P_rap->global_num_rows);
    ASSERT_EQ(P->global_num_cols, P_rap->global_num_cols);
    ASSERT_EQ(P->local_num_rows, P_rap->local_num_rows);
    ASSERT_EQ(P->on_proc_num_cols, P_rap->on_proc_num_cols);
    ASSERT_EQ(P->off_proc_num_cols, P_rap->off_proc_num_cols);

    P->on_proc->sort();
    P->on_proc->move_diag();
    P->off_proc->sort();
    P_rap->on_proc->sort();
    P_rap->on_proc->move_diag();
    P_rap->off_proc->sort();

    ASSERT_EQ(P->on_proc->idx1[0], P_rap->on_proc->idx1[0]);
    ASSERT_EQ(P->off_proc->idx1[0], P_rap->off_proc->idx1[0]);
    for (int i = 0; i < P->local_num_rows; i++)
    {
        ASSERT_EQ(P->on_proc->idx1[i+1], P_rap->on_proc->idx1[i+1]);
        start = P->on_proc->idx1[i];
        end = P->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            ASSERT_EQ(P->on_proc->idx2[j], P_rap->on_proc->idx2[j]);
            if (P_rap->on_proc->format() == CSR)
                ASSERT_NEAR(P->on_proc->vals[j], P_rap->on_proc->vals[j], 1e-06);
        }

        ASSERT_EQ(P->off_proc->idx1[i+1], P_rap->off_proc->idx1[i+1]);
        start = P->off_proc->idx1[i];
        end = P->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            ASSERT_EQ(P->off_proc->idx2[j], P_rap->off_proc->idx2[j]);
            if (P_rap->off_proc->format() == CSR)
                ASSERT_NEAR(P->off_proc->vals[j], P_rap->off_proc->vals[j], 1e-06);
        }
    }
    
    ASSERT_EQ(P->comm->send_data->num_msgs, P_rap->comm->send_data->num_msgs);
    ASSERT_EQ(P->comm->send_data->size_msgs, P_rap->comm->send_data->size_msgs);
    ASSERT_EQ(P->comm->recv_data->num_msgs, P_rap->comm->recv_data->num_msgs);
    ASSERT_EQ(P->comm->recv_data->size_msgs, P_rap->comm->recv_data->size_msgs);

    // TODO -- can compare send indices (need to sort procs)
    // Add idx of each proc to vector
    // Go through P_rap msgs... find idx in vector, and compare to P indices
    
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    std::vector<int> proc_idx(num_procs);
    ASSERT_EQ(P->comm->send_data->indptr[0], P_rap->comm->send_data->indptr[0]);
    for (int i = 0; i < P->comm->send_data->num_msgs; i++)
    {
        int proc = P->comm->send_data->procs[i];
        proc_idx[proc] = i;
    }
    for (int i = 0; i < P->comm->send_data->num_msgs; i++)
    {
        int proc = P_rap->comm->send_data->procs[i];
        int idx = proc_idx[proc];

        start = P->comm->send_data->indptr[idx];
        end = P->comm->send_data->indptr[idx+1];
        int start_rap = P_rap->comm->send_data->indptr[i];
        int end_rap = P_rap->comm->send_data->indptr[i+1];
        ASSERT_EQ(end - start, (end_rap - start_rap));
        int size = end - start;
        for (int j = 0; j < size; j++)
        {
            ASSERT_EQ(P->comm->send_data->indices[start + j],
                      P_rap->comm->send_data->indices[start_rap + j]);
        }
    }
    ASSERT_EQ(P->comm->recv_data->indptr[0], P_rap->comm->recv_data->indptr[0]);
    for (int i = 0; i < P->comm->recv_data->num_msgs; i++)
    {
        ASSERT_EQ(P->comm->recv_data->indptr[i+1], P_rap->comm->recv_data->indptr[i+1]);
        start = P->comm->recv_data->indptr[i];
        end = P->comm->recv_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            ASSERT_EQ(P->comm->recv_data->indices[j], P_rap->comm->recv_data->indices[j]);
        }
    }
}


TEST(TestParRugeStuben, TestsInRuge_Stuben)
{ 

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    ParCSRMatrix* A0;
    ParVector x0;
    ParVector b0;
    ParVector tmp0;
    Multilevel* ml;
    int first_row, first_col;
    FILE* f;
    int cf;
    std::vector<int> splitting;
    std::vector<int> off_splitting;
    std::vector<int> proc_sizes(num_procs);

    // Create fine grid problem
    A0 = readParMatrix("../../tests/rss_laplace_A0.mtx", MPI_COMM_WORLD, 1, 1);
    x0.resize(A0->global_num_rows, A0->local_num_rows, A0->partition->first_local_row);
    tmp0.resize(A0->global_num_rows, A0->local_num_rows, A0->partition->first_local_row);
    b0.resize(A0->global_num_rows, A0->local_num_rows, A0->partition->first_local_row);
    x0.set_const_value(1.0);
    A0->mult(x0, b0);
    x0.set_const_value(0.0);
    
    // Create strength and compare 
    ParCSRBoolMatrix* S0 = A0->strength(0.25);
    ParCSRMatrix* S0_py = readParMatrix("../../tests/rss_laplace_S0.mtx", MPI_COMM_WORLD, 1, 1);
    compare(S0, S0_py);
    delete S0_py;

    split_cljp(S0, splitting, off_splitting);
    compare_splitting(S0->local_num_rows, splitting, "../../tests/rss_laplace_cf0.txt");

    ParCSRMatrix* P0 = direct_interpolation(A0, S0, splitting, off_splitting);
    MPI_Allgather(&P0->on_proc_num_cols, 1, MPI_INT, proc_sizes.data(), 1, MPI_INT,
            MPI_COMM_WORLD);
    first_col = 0;
    for (int i = 0; i < rank; i++)
    {
        first_col += proc_sizes[i];
    }

    ParCSRMatrix* P0_py = readParMatrix("../../tests/rss_laplace_P0.mtx", MPI_COMM_WORLD, 1, 0,
            P0->local_num_rows, P0->on_proc_num_cols, P0->partition->first_local_row, first_col);
    compare(P0, P0_py);
    delete P0_py;
    delete S0;

    // P_T*A*P == form coarse grid operator
    first_row = first_col;
    ParCSRMatrix* AP0 = A0->mult(P0);
    ParCSCMatrix* P0_csc = new ParCSCMatrix(P0);
    ParCSRMatrix* A1 = AP0->mult_T(P0_csc);
    A1->comm = new ParComm(A1->partition, A1->off_proc_column_map, A1->on_proc_column_map);
    ParCSRMatrix* A1_py = readParMatrix("../../tests/rss_laplace_A1.mtx", MPI_COMM_WORLD, 1, 0,
            P0->on_proc_num_cols, P0->on_proc_num_cols, first_row, first_col);
    compare(A1, A1_py);
    delete AP0;
    delete P0_csc;
    ParVector x1(A1->global_num_rows, A1->local_num_rows, A1->partition->first_local_row);
    ParVector tmp1(A1->global_num_rows, A1->local_num_rows, A1->partition->first_local_row);
    ParVector b1(A1->global_num_rows, A1->local_num_rows, A1->partition->first_local_row);

    // Compare Strength
    ParCSRBoolMatrix* S1 = A1->strength(0.25);
    ParCSRMatrix* S1_py = readParMatrix("../../tests/rss_laplace_S1.mtx", MPI_COMM_WORLD, 1, 0, 
            P0->on_proc_num_cols, P0->on_proc_num_cols, first_row, first_col);
    compare(S1, S1_py);
    delete S1_py;

    // Read in splitting, create prologation operator
    split_cljp(S1, splitting, off_splitting);
    compare_splitting(S1->local_num_rows, splitting, "../../tests/rss_laplace_cf1.txt");
    S1->comm->communicate(splitting);
    for (int i = 0; i < S1->off_proc_num_cols; i++)
    {
        ASSERT_EQ(off_splitting[i], S1->comm->recv_data->int_buffer[i]);
    }

    ParCSRMatrix* P1 = direct_interpolation(A1, S1, splitting, off_splitting);
    MPI_Allgather(&P1->on_proc_num_cols, 1, MPI_INT, proc_sizes.data(), 1, MPI_INT,
            MPI_COMM_WORLD);
    first_col = 0;
    for (int i = 0; i < rank; i++)
    {
        first_col += proc_sizes[i];
    }
    ParCSRMatrix* P1_py = readParMatrix("../../tests/rss_laplace_P1.mtx", MPI_COMM_WORLD, 1, 0,
            P0->on_proc_num_cols, P1->on_proc_num_cols, first_row, first_col);
    compare(P1, P1_py);
    delete S1;
    
    // P_T*A*P == form coarse grid operator
    ParCSRMatrix* AP1 = A1->mult(P1);
    ParCSCMatrix* P1_csc = new ParCSCMatrix(P1); 
    ParCSRMatrix* A2 = AP1->mult_T(P1_csc);
    A2->comm = new ParComm(A2->partition, A2->off_proc_column_map, A2->on_proc_column_map);
    delete P1_csc;
    ParVector x2(A2->global_num_rows, A2->local_num_rows, A2->partition->first_local_row);
    ParVector tmp2(A2->global_num_rows, A2->local_num_rows, A2->partition->first_local_row);
    ParVector b2(A2->global_num_rows, A2->local_num_rows, A2->partition->first_local_row);

    ParCSRMatrix* A2_py = readParMatrix("../../tests/rss_laplace_A2.mtx", MPI_COMM_WORLD, 1, 0,
            P1->on_proc_num_cols, P1->on_proc_num_cols, first_col, first_col);
    compare(A2, A2_py);
    delete AP1;
    delete A1_py;
    delete P1_py;
    delete A2_py;

    A0->residual(x0, b0, tmp0);
    double orig_norm = tmp0.norm(2);
    if (rank == 0) printf("Orig Norm = %e\n", orig_norm);

    for (int i = 0; i < 10; i++)
    {
        // Level 0
        relax(A0, x0, b0, tmp0);
        A0->residual(x0, b0, tmp0);
        P0->mult_T(tmp0, b1);

        // Level 1
        x1.set_const_value(0.0);
        relax(A1, x1, b1, tmp1);
        A1->residual(x1, b1, tmp1);
        P1->mult_T(tmp1, b2);

        // Coarsest level
        x2.set_const_value(0.0);
        relax(A2, x2, b2, tmp2);

        // Level 1
        P1->mult(x2, tmp1);
        for (int i = 0; i < A1->local_num_rows; i++)
        {
            x1[i] += tmp1[i];
        }
        relax(A1, x1, b1, tmp1);

        // Level 0
        P0->mult(x1, tmp0);
        for (int i = 0; i < A0->local_num_rows; i++)
        {
            x0[i] += tmp0[i];
        }
        relax(A0, x0, b0, tmp0);

        A0->residual(x0, b0, tmp0);
        double bnorm = tmp0.norm(2);
        if (rank == 0) printf("Rnorm = %e, RelResid = %e\n", bnorm, bnorm/orig_norm);
    }

    delete A2;
    delete P1;
    delete A1;
    delete P0;
    delete A0;

} // end of TEST(TestParRugeStuben, TestsInRuge_Stuben) //
