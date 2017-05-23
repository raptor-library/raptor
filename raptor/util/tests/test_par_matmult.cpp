#include <assert.h>
#include <math.h>
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/matrix_IO.hpp"

using namespace raptor;

// Compare A_solution to computed A_par
void compare(ParCSCMatrix* A_sol, ParCSCMatrix* A_par)
{
    std::vector<double> col_vals;
    std::vector<int> next;
    if (A_sol->local_num_rows)
    {
        col_vals.resize(A_sol->local_num_rows, 0);
        next.resize(A_sol->local_num_rows);
    }

    int head, length;
    int col_start, col_end;
    int row;
    int col_start_sol, col_end_sol;

    assert(A_sol->global_num_rows == A_par->global_num_rows);
    assert(A_sol->local_num_rows == A_par->local_num_rows);
    assert(A_sol->first_local_row == A_par->first_local_row);
    assert(A_sol->global_num_cols == A_par->global_num_cols);
    assert(A_sol->local_num_cols == A_par->local_num_cols);
    assert(A_sol->first_local_col == A_par->first_local_col);

    for (int col = 0; col < A_sol->local_num_cols; col++)
    {
        col_start_sol = A_sol->on_proc->idx1[col];
        col_end_sol = A_sol->on_proc->idx1[col+1];
        for (int j = col_start_sol; j < col_end_sol; j++)
        {
            row = A_sol->on_proc->idx2[j];
            col_vals[row] = A_sol->on_proc->vals[j];
            next[row] = head;
            head = row;
            length++;
        }

        col_start = A_par->on_proc->idx1[col];
        col_end = A_par->on_proc->idx1[col+1];

        //assert(col_start == col_start_sol);
        //assert(col_end == col_end_sol);

        for (int j = col_start; j < col_end; j++)
        {
            row = A_par->on_proc->idx2[j];
            assert(fabs(A_par->on_proc->vals[j] - col_vals[row]) < 1e-8);
        }

        for (int j = 0; j < length; j++)
        {
            col_vals[head] = 0;
            head = next[head];
        }
    }

    for (int col = 0; col < A_sol->off_proc_num_cols; col++)
    {
        col_start_sol = A_sol->off_proc->idx1[col];
        col_end_sol = A_sol->off_proc->idx1[col+1];
        for (int j = col_start_sol; j < col_end_sol; j++)
        {
            row = A_sol->off_proc->idx2[j];
            col_vals[row] = A_sol->off_proc->vals[j];
            next[row] = head;
            head = row;
            length++;
        }

        col_start = A_par->off_proc->idx1[col];
        col_end = A_par->off_proc->idx1[col+1];
                
        //assert(col_start == col_start_sol);
        //assert(col_end == col_end_sol);

        for (int j = col_start; j < col_end; j++)
        {
            row = A_par->off_proc->idx2[j];
            assert(fabs(A_par->off_proc->vals[j] - col_vals[row]) < 1e-8);
        }
        for (int j = 0; j < length; j++)
        {
            col_vals[head] = 0;
            head = next[head];
        }
    }  
}

// Compare A_solution to computed A_par
void compare(ParCSRMatrix* A_sol, ParCSRMatrix* A_par)
{
    std::vector<double> row_vals;
    std::vector<int> next;

    int head, length;
    int row_start, row_end;
    int col;
    int row_start_sol, row_end_sol;

    assert(A_sol->global_num_rows == A_par->global_num_rows);
    assert(A_sol->local_num_rows == A_par->local_num_rows);
    assert(A_sol->first_local_row == A_par->first_local_row);
    assert(A_sol->global_num_cols == A_par->global_num_cols);
    assert(A_sol->local_num_cols == A_par->local_num_cols);
    assert(A_sol->first_local_col == A_par->first_local_col);

    if (A_sol->local_num_cols)
    {
        row_vals.resize(A_sol->local_num_cols, 0);
        next.resize(A_sol->local_num_cols);
    }

    for (int row = 0; row < A_sol->local_num_rows; row++)
    {
        head = -2;
        length = 0;

        row_start_sol = A_sol->on_proc->idx1[row];
        row_end_sol = A_sol->on_proc->idx1[row+1];
        for (int j = row_start_sol; j < row_end_sol; j++)
        {
            col = A_sol->on_proc->idx2[j];
            row_vals[col] = A_sol->on_proc->vals[j];
            next[col] = head;
            head = col;
            length++;
        }

        row_start = A_par->on_proc->idx1[row];
        row_end = A_par->on_proc->idx1[row+1];
        
        //assert(row_start == row_start_sol);
        //assert(row_end == row_end_sol);

        for (int j = row_start; j < row_end; j++)
        {
            col = A_par->on_proc->idx2[j];
            if (fabs(A_par->on_proc->vals[j] - row_vals[col]) > 1e-8)
                printf("Apar %e, Asol %e, Diff %e\n", A_par->on_proc->vals[j],
                        row_vals[col], fabs(A_par->on_proc->vals[j] - row_vals[col]));
            //assert(fabs(A_par->on_proc->vals[j] - row_vals[col]) < 1e-8);
        }

        for (int j = 0; j < length; j++)
        {
            row_vals[head] = 0;
            head = next[head];
        }
    }

    if (A_sol->off_proc_num_cols)
    {
        row_vals.resize(A_sol->off_proc_num_cols, 0);
        next.resize(A_sol->off_proc_num_cols);
    }

    for (int row = 0; row < A_sol->local_num_rows; row++)
    {
        head = -2;
        length = 0;

        row_start_sol = A_sol->off_proc->idx1[row];
        row_end_sol = A_sol->off_proc->idx1[row+1];
        for (int j = row_start_sol; j < row_end_sol; j++)
        {
            col = A_sol->off_proc->idx2[j];
            row_vals[col] = A_sol->off_proc->vals[j];
            next[col] = head;
            head = col;
            length++;
        }

        row_start = A_par->off_proc->idx1[row];
        row_end = A_par->off_proc->idx1[row+1];

        //assert(row_start == row_start_sol);
        //assert(row_end == row_end_sol);

        for (int j = row_start; j < row_end; j++)
        {
            col = A_par->off_proc->idx2[j];
            //assert(fabs(A_par->off_proc->vals[j] - row_vals[col]) < 1e-8);
        }

        for (int j = 0; j < length; j++)
        {
            row_vals[head] = 0;
            head = next[head];
        }
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    ParCSRMatrix* Acsr = readParMatrix("/Users/abienz/Documents/Parallel/raptor_topo/build/raptor/util/tests/testA.mtx", MPI_COMM_WORLD, 1, 1);
    ParCSRMatrix* Pcsr = readParMatrix("/Users/abienz/Documents/Parallel/raptor_topo/build/raptor/util/tests/testP.mtx", MPI_COMM_WORLD, 1, 0);
    ParCSCMatrix* Acsc = new ParCSCMatrix(Acsr);
    ParCSCMatrix* Pcsc = new ParCSCMatrix(Pcsr);

    ParCSRMatrix* AP_sol_csr = readParMatrix("testAP.mtx", MPI_COMM_WORLD, 1, 0, 
            Acsr->local_num_rows, Pcsr->local_num_cols, 
            Acsr->first_local_row, Pcsr->first_local_col);
    ParCSRMatrix* Ac_sol_csr = readParMatrix("testAc.mtx", MPI_COMM_WORLD, 1, 1,
            Pcsr->local_num_cols, Pcsr->local_num_cols,
            Pcsr->first_local_col, Pcsr->first_local_col);
    ParCSCMatrix* AP_sol_csc = new ParCSCMatrix(AP_sol_csr);
    ParCSCMatrix* Ac_sol_csc = new ParCSCMatrix(Ac_sol_csr);

    ParCSRMatrix* APcsr = Acsr->mult(Pcsr);
    compare(AP_sol_csr, APcsr);


    delete Acsr;
    delete Pcsr;
    delete Acsc;
    delete Pcsc;
    delete AP_sol_csr;
    delete Ac_sol_csr;
    delete AP_sol_csc;
    delete Ac_sol_csc;

    delete APcsr;

    MPI_Finalize();
}

