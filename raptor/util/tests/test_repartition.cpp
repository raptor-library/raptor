#include <assert.h>
#include <math.h>
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "util/linalg/external/repartition.hpp"

using namespace raptor;

void compare(Vector& b, ParVector& b_par)
{
    double b_norm = b.norm(2);
    double b_par_norm = b_par.norm(2);

    assert(fabs(b_norm - b_par_norm) < 1e-06);

    Vector& b_par_lcl = b_par.local;
    for (int i = 0; i < b_par.local_n; i++)
    {
        assert(fabs(b_par_lcl[i] - b[i+b_par.first_local]) < 1e-06);
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Create Sequential Matrix A on each process
    ParCSRMatrix* A_tmp = readParMatrix("../../../../examples/1138_bus.mtx", MPI_COMM_WORLD, 
            true, 1);
    //int* proc_part = ptscotch_partition(A_tmp);
    int* proc_part = new int[A_tmp->local_num_rows];
    for (int i  = 0; i < A_tmp->local_num_rows; i++)
    {
        proc_part[i] = i % num_procs;
    } 
    
    ParCSRMatrix* A = repartition_matrix(A_tmp, proc_part);

    delete[] proc_part;
    delete A_tmp;
    delete A;

    MPI_Finalize();
}

