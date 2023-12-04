// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor/raptor.hpp"
#include "raptor/tests/par_compare.hpp"

using namespace raptor;


int argc;
char **argv;

int main(int _argc, char** _argv)
{
    MPI_Init(&_argc, &_argv);
    
    ::testing::InitGoogleTest(&_argc, _argv);
    argc = _argc;
    argv = _argv;
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //


TEST(ParSparsifyTest, TestsInMultilevel)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FILE* f;
    int cf;

    ParCSRMatrix* A;
    std::vector<int> states;
    std::vector<double> weights;
    ParCSRMatrix* S;
    ParCSRMatrix* P;
    ParCSRMatrix* I;
    ParCSRMatrix* Ac;
    ParCSRMatrix* Ac_rap;

    const char* A0_fn = "../../../../test_data/rss_A0.pm";
    const char* weight_fn = "../../../../test_data/weights.txt";
    const char* cf0_fn = "../../../../test_data/rss_cf0.txt";
    const char* A1_fn = "../../../../test_data/rss_A1.pm";
    const char* A1_hgal_fn = "../../../../test_data/rss_A1_hgal.pm";

    A = readParMatrix(A0_fn);
    S = A->strength(Classical, 0.25);

    f = fopen(weight_fn, "r");
    weights.resize(S->local_num_rows);
    for (int i = 0; i < S->partition->first_local_row; i++)
    {
        fscanf(f, "%lf\n", &weights[0]);
    }
    for (int i = 0; i < S->local_num_rows; i++)
    {
        fscanf(f, "%lf\n", &weights[i]);
    }
    fclose(f);

    states.resize(A->local_num_rows);
    f = fopen(cf0_fn, "r");
    for (int i = 0; i < S->partition->first_local_row; i++)
    {
        fscanf(f, "%d\n", &states[0]);
    }
    for (int i = 0; i < S->local_num_rows; i++)
    {
        fscanf(f, "%d\n", &states[i]);
    }
    fclose(f);
    S->comm->communicate(states);

    P = mod_classical_interpolation(A, S, states, S->comm->recv_data->int_buffer);

    std::vector<int> first_rows(num_procs+1);
    first_rows[0] = 0;
    MPI_Allgather(&P->on_proc_num_cols, 1, MPI_INT, &first_rows[1], 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < num_procs; i++)
    {
        first_rows[i+1] += first_rows[i];
    }

    ParCSRMatrix* AP = A->mult(P);
    Ac = AP->mult_T(P);
    Ac->comm = new ParComm(Ac->partition, Ac->off_proc_column_map, Ac->on_proc_column_map);
    Ac_rap = readParMatrix(A1_hgal_fn, Ac->local_num_rows, Ac->on_proc_num_cols, 
            first_rows[rank], first_rows[rank]);


    I = new ParCSRMatrix(P->partition, P->global_num_rows, P->global_num_cols,
            P->local_num_rows, P->on_proc_num_cols, 0);
    I->on_proc->idx1[0] = 0;
    I->off_proc->idx1[0] = 0;
    int ctr = 0;
    for (int i = 0; i < P->local_num_rows; i++)
    {
        if (states[i])
        {
            I->on_proc->idx2.push_back(ctr++);
            I->on_proc->vals.push_back(1.0);
        }
        I->on_proc->idx1[i+1] = I->on_proc->idx2.size();
        I->off_proc->idx1[i+1] = 0;
    }
    I->on_proc->nnz = I->on_proc->idx2.size();
    I->off_proc->nnz = 0;
    I->finalize();

    printf("%d, %d\n", Ac->off_proc_num_cols, Ac_rap->off_proc_num_cols);
    sparsify(A, P, I, AP, Ac, 0.1);
    printf("%d, %d\n", Ac->off_proc_num_cols, Ac_rap->off_proc_num_cols);
    //compare(Ac, Ac_rap);

    delete AP;
    delete P;
    delete I;
    delete Ac;
    delete Ac_rap;
    delete S;
    delete A;

} // end of TEST(ParAMGTest, TestsInMultilevel) //

