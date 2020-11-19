#include <assert.h>
#include "raptor.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Needed for partial inner products
    int inner_color, root_color, inner_root, procs_in_group, part_global;
    double frac;
    MPI_Comm inner_comm = MPI_COMM_NULL;
    MPI_Comm root_comm = MPI_COMM_NULL;

    if (argc < 2) {
        printf("Include fraction for partial inner product\n");
        exit(-1);
    }

    frac = atof(argv[1]);

    // Setup problem to solve
    int grid[3] = {10, 10, 10};
    double* stencil = laplace_stencil_27pt();
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 3);
    
    ParVector x_part(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector x_true(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector b(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    std::vector<double> residuals_true;
    std::vector<double> residuals_part;

    x_true.set_const_value(1.0);
    A->mult(x_true, b);
    x_true.set_const_value(0.0);
    BiCGStab(A, x_true, b, residuals_true);

    MPI_Barrier(MPI_COMM_WORLD);

    // Test half
    x_part.set_const_value(1.0);
    A->mult(x_part, b);
    x_part.set_const_value(0.0);
    PI_BiCGStab(A, x_part, b, residuals_part, inner_comm, root_comm, frac, inner_color, root_color, inner_root, procs_in_group,
                part_global);

    MPI_Barrier(MPI_COMM_WORLD);
    
    // Write out residuals to file
    if (rank == 0) {
        FILE *f;
        const char *prob = "laplacian";
        const char *start_fname = "_PartInner_";
        const char *end_fname = "_BiCGStab_Res.txt";
        char fname_buffer[512];
        sprintf(fname_buffer, "%s%s%f%s", prob, start_fname, frac, end_fname);
        f = fopen(fname_buffer, "w");
        fprintf(f, "Grid = {10, 10, 10} Laplacian 3d 27-pt Stencil %d x %d\n", A->global_num_rows, A->global_num_cols);
        for (int i=0; i<residuals_part.size(); i++) {
            fprintf(f, "%lf \n", residuals_part[i]);
        }
        fclose(f);
       
        sprintf(fname_buffer, "%s%s", prob, end_fname);
        f = fopen(fname_buffer, "w");
        fprintf(f, "Grid = {10, 10, 10} Laplacian 3d 27-pt Stencil %d x %d\n", A->global_num_rows, A->global_num_cols);
        for (int i=0; i<residuals_true.size(); i++) {
            fprintf(f, "%lf\n", residuals_true[i]);
        }
        fclose(f);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) printf("Testing Contiguous Solution\n");
    for (int i = 0; i < x_true.local_n; i++) {
        assert(fabs(x_true.local[i] - x_part.local[i]) < 1e-05);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Comm_free(&inner_comm);
    MPI_Comm_free(&root_comm);

    delete[] stencil;
    delete A;

    MPI_Finalize();

    return 0;
}
