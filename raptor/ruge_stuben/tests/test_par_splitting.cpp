#include <assert.h>
#include "mpi.h"
#include "gallery/stencil.hpp"
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "ruge_stuben/par_cf_splitting.hpp"
#include <iostream>
#include <fstream>

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FILE* f;
    std::vector<int> states;
    std::vector<int> off_proc_states;
    int cf;

    ParCSRMatrix* S;

    // TEST LEVEL 0
    S = readParMatrix("../../../../test_data/rss_S0.mtx", MPI_COMM_WORLD, 1, 1);

    f = fopen("../../../../test_data/weights.txt", "r");
    std::vector<double> weights(S->local_num_rows);
    for (int i = 0; i < S->partition->first_local_row; i++)
    {
        fscanf(f, "%lf\n", &weights[0]);
    }
    for (int i = 0; i < S->local_num_rows; i++)
    {
        fscanf(f, "%lf\n", &weights[i]);
    }
    fclose(f);
    split_cljp(S, states, off_proc_states, weights.data());
    
    f = fopen("../../../../test_data/rss_cf0", "r");
    for (int i = 0; i < S->partition->first_local_row; i++)
    {
        fscanf(f, "%d\n", &cf);
    }
    for (int i = 0; i < S->local_num_rows; i++)
    {
        fscanf(f, "%d\n", &cf);
        assert(cf == states[i]);
    }
    fclose(f);

    delete S;

    // TEST LEVEL 1
    S = readParMatrix("../../../../test_data/rss_S1.mtx", MPI_COMM_WORLD, 1, 0);

    f = fopen("../../../../test_data/weights.txt", "r");
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
    split_cljp(S, states, off_proc_states, weights.data());
    
    f = fopen("../../../../test_data/rss_cf1", "r");
    for (int i = 0; i < S->partition->first_local_row; i++)
    {
        fscanf(f, "%d\n", &cf);
    }
    for (int i = 0; i < S->local_num_rows; i++)
    {
        fscanf(f, "%d\n", &cf);
        assert(cf == states[i]);
    }
    fclose(f);

    delete S;


    MPI_Finalize();

    return 0;
}
