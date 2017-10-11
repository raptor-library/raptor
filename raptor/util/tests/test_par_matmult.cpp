#include <assert.h>
#include <math.h>
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_matrix_IO.hpp"

using namespace raptor;

// Compare A_solution to computed A_par
void compare(ParCSRMatrix* A0, ParCSRMatrix* A1)
{
    int start, end;
    int col;

    assert(A0->global_num_rows == A1->global_num_rows);
    assert(A0->local_num_rows == A1->local_num_rows);
    assert(A0->global_num_cols == A1->global_num_cols);
    assert(A0->on_proc_num_cols == A1->on_proc_num_cols);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int row = 0; row < A0->local_num_rows; row++)
    {
        start = A1->on_proc->idx1[row];
        end = A1->on_proc->idx1[row+1];
        assert(A0->on_proc->idx1[row] == start);
        assert(A0->on_proc->idx1[row+1] == end);

        for (int j = start; j < end; j++)
        {
            assert(A0->on_proc->idx2[j] == A1->on_proc->idx2[j]);
            assert(fabs(A0->on_proc->vals[j] - A1->on_proc->vals[j]) < 1e-08);
        }
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    char* fname = "/Users/abienz/Documents/Parallel/raptor_topo/raptor/util/tests/testA.mtx";
    ParCSRMatrix* Acsr = readParMatrix(fname, MPI_COMM_WORLD, 1, 1);

    fname = "/Users/abienz/Documents/Parallel/raptor_topo/raptor/util/tests/testP.mtx";
    ParCSRMatrix* Pcsr = readParMatrix(fname, MPI_COMM_WORLD, 1, 0);
    ParCSCMatrix* Pcsc = new ParCSCMatrix(Pcsr);

    fname = "/Users/abienz/Documents/Parallel/raptor_topo/raptor/util/tests/testAP.mtx";
    ParCSRMatrix* AP_sol_csr = readParMatrix(fname, MPI_COMM_WORLD, 1, 0, 
            Acsr->local_num_rows, Pcsr->on_proc_num_cols, 
            Acsr->partition->first_local_row, Pcsr->partition->first_local_col);

    fname = "/Users/abienz/Documents/Parallel/raptor_topo/raptor/util/tests/testAc.mtx";
    ParCSRMatrix* Ac_sol_csr = readParMatrix(fname, MPI_COMM_WORLD, 1, 0,
            Pcsr->on_proc_num_cols, Pcsr->on_proc_num_cols,
            Pcsr->partition->first_local_col, Pcsr->partition->first_local_col);

    AP_sol_csr->on_proc->sort();
    AP_sol_csr->off_proc->sort();
    Ac_sol_csr->on_proc->sort();
    Ac_sol_csr->off_proc->sort();

    // Test CSR <- CSR.mult(CSR)
    ParCSRMatrix* APcsr = Acsr->mult(Pcsr);
    APcsr->on_proc->sort();
    APcsr->off_proc->sort();
    compare(AP_sol_csr, APcsr);
    delete APcsr;

    APcsr = Acsr->tap_mult(Pcsr);
    APcsr->on_proc->sort();
    APcsr->off_proc->sort();
    compare(AP_sol_csr, APcsr);
    delete APcsr;

    // Test CSR <- CSR.mult_T(CSC)
    ParCSRMatrix* Accsr = AP_sol_csr->mult_T(Pcsc);
    Accsr->on_proc->sort();
    Accsr->off_proc->sort();
    compare(Ac_sol_csr, Accsr);
    delete Accsr;

    Accsr = AP_sol_csr->tap_mult_T(Pcsc);
    Accsr->on_proc->sort();
    Accsr->off_proc->sort();
    compare(Ac_sol_csr, Accsr);
    delete Accsr;

    delete Acsr;
    delete Pcsr;
    delete Pcsc;
    delete AP_sol_csr;
    delete Ac_sol_csr;

    MPI_Finalize();
}

