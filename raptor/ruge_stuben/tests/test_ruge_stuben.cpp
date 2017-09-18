#include <assert.h>

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/laplacian27pt.hpp"

#include "ruge_stuben/cf_splitting.hpp"
#include "ruge_stuben/interpolation.hpp"

#include "multilevel/multilevel.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double* stencil = laplace_stencil_27pt();
    int n = 10;
    if (argc > 1)
    {
        n = atoi(argv[1]);
    }
    std::vector<int> grid(3, n);

    ParCSRMatrix* A = par_stencil_grid(stencil, grid.data(), 3);
    A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map);
    ParVector x(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector b(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);

    Multilevel* ml = new Multilevel(A);
    ml->solve(x, b);

    delete ml;


    ParCSRMatrix* S = A->strength(0.0);
    std::vector<int> states;
    std::vector<int> off_proc_states;
    cf_splitting(S, states, off_proc_states);
    ParCSRMatrix* P = direct_interpolation(A, S, states, off_proc_states);

    ParCSRMatrix* AP = A->mult(P);
    ParCSCMatrix* P_csc = new ParCSCMatrix(P);
    ParCSRMatrix* Ac = AP->mult_T(P_csc);
    Ac->comm = new ParComm(Ac->partition,
            Ac->off_proc_column_map,
            Ac->on_proc_column_map);

    delete Ac;
    delete P_csc;
    delete AP;
    delete P;
    delete S;
    delete A;
    delete[] stencil;

    MPI_Finalize();
    return 0;
}
