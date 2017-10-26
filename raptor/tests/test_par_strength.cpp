#include <assert.h>

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "tests/par_compare.hpp"

using namespace raptor;


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    ParCSRMatrix* A;
    ParCSRMatrix* S;
    ParCSRBoolMatrix* S_rap;

    A = readParMatrix("../../../test_data/rss_A0.mtx", MPI_COMM_WORLD, 1, 1);
    S = readParMatrix("../../../test_data/rss_S0.mtx", MPI_COMM_WORLD, 1, 1);
    S_rap = A->strength(0.25);
    compare(S, S_rap);
    delete A;
    delete S;
    delete S_rap;

    A = readParMatrix("../../../test_data/rss_A1.mtx", MPI_COMM_WORLD, 1, 0);
    S = readParMatrix("../../../test_data/rss_S1.mtx", MPI_COMM_WORLD, 1, 0);
    S_rap = A->strength(0.25);
    compare(S, S_rap);
    delete A;
    delete S;
    delete S_rap;

    MPI_Finalize();
}
