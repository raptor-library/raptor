#include <assert.h>

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/par_matrix.hpp"
#include "gallery/stencil.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/diffusion.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Form 100x100 A_orig
    double eps = 0.001;
    double theta = M_PI / 8.0;
    int grid[2] = {10, 10};
    double* stencil = diffusion_stencil_2d(eps, theta);
    ParCSRMatrix* A_orig = par_stencil_grid(stencil, grid, 2);
    ParVector x_orig(A_orig->partition->global_num_cols, 
            A_orig->partition->local_num_cols, 
            A_orig->partition->first_local_col);
    ParVector b_orig(A_orig->partition->global_num_rows, 
            A_orig->partition->local_num_rows, 
            A_orig->partition->first_local_row);
    
    int orig_n_rows = A_orig->partition->global_num_rows;

    // Form condensed matrix A
    ParCSRMatrix* A = new ParCSRMatrix(10*orig_n_rows, 10*orig_n_rows);
    ParVector x(A->partition->global_num_cols, 
            A->partition->local_num_cols, 
            A->partition->first_local_col);
    ParVector b(A->partition->global_num_rows, 
            A->partition->local_num_rows, 
            A->partition->first_local_row);

    x.set_const_value(1.0);
    x_orig.set_const_value(1.0);


    printf("localrowsorig: %d, localrows = %d\n", A_orig->partition->local_num_rows,
            A->partition->local_num_rows);

    printf("globalrowsorig: %d, globalrows = %d\n", A_orig->partition->global_num_rows,
            A->partition->global_num_rows);

    // Copy on_proc and off_proc matrices from A_orig
    A->on_proc->copy((CSRMatrix*) A_orig->on_proc);
    A->off_proc->copy((CSRMatrix*) A_orig->off_proc);

    // Map on_proc columns and rows to cols/rows in A
    // (orig idx times 10)
    for (int i = 0; i < A_orig->local_num_cols; i++)
    {
        A->on_proc->col_list.push_back(i*10);
    }
    for (int i = 0; i < A_orig->local_num_rows; i++)
    {
        A->on_proc->row_list.push_back(i*10);
    }

    // Map off_proc rows to rows in A
    for (int i = 0; i < A_orig->local_num_rows; i++)
    {
        A->off_proc->row_list.push_back(i*10);
    }    

    A->off_proc_column_map = A->off_proc->get_col_list();
    A->off_proc_num_cols = A->off_proc_column_map.size();

    A->comm = new ParComm(A->partition, A->off_proc_column_map);
    A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map);

    A_orig->mult(x_orig, b_orig);
    //A->mult(x, b);
    A->tap_mult(x, b);

    if (rank == 0)
    {
        printf("Borig:\n");
        b_orig.local.print();
        printf("B:\n");
        b.local.print();
        printf("Blocalsize = %d\n", b.local.size());
    }

    delete[] stencil;
    delete A_orig;
    delete A;

    MPI_Finalize();
}


