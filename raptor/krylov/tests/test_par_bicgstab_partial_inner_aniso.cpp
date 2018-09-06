#include <assert.h>
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "krylov/par_bicgstab.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_stencil.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Needed for partial inner products
    int first_root = 0, second_root = 0, color = 0, part_global;
    MPI_Comm contig_comm = MPI_COMM_NULL;
    MPI_Comm striped_comm = MPI_COMM_NULL;

    // Setup problem to solve
    int grid[2] = {50, 50};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    //double* stencil = diffusion_stencil_2d(0.1, M_PI/4.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    
    ParVector x_contig(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector x_striped(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector x_true(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector b(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    aligned_vector<double> residuals_true;
    aligned_vector<double> residuals_contig;
    aligned_vector<double> residuals_striped;

    x_true.set_const_value(1.0);
    A->mult(x_true, b);
    x_true.set_const_value(0.0);
    BiCGStab(A, x_true, b, residuals_true);

    MPI_Barrier(MPI_COMM_WORLD);

    // Test contiguous first half
    //create_partial_inner_comm(inner_comm, color, first_root, second_root, part_global, 0);
    x_contig.set_const_value(1.0);
    A->mult(x_contig, b);
    x_contig.set_const_value(0.0);
    PI_BiCGStab(A, x_contig, b, residuals_contig, contig_comm, color, first_root, second_root, part_global, 0);

    MPI_Barrier(MPI_COMM_WORLD);

    // Test striped even procs
    //create_partial_inner_comm(striped_comm, color, first_root, second_root, part_global, 1);
    x_striped.set_const_value(1.0);
    A->mult(x_striped, b);
    x_striped.set_const_value(0.0);
    PI_BiCGStab(A, x_striped, b, residuals_striped, striped_comm, color, first_root, second_root, part_global, 1);

    MPI_Barrier(MPI_COMM_WORLD);

    // Write out residuals to file
    if (rank == 0) {
       FILE *f;
       f = fopen("PartInner_Contig_BiCGStab_Res.txt", "w");
       fprintf(f, "Grid = {50,50} Diffusion 2d Stencil (0.001, pi/8)  %d x %d\n", A->global_num_rows, A->global_num_cols);
       for (int i=0; i<residuals_contig.size(); i++) {
           fprintf(f, "%lf \n", residuals_contig[i]);
       }
       fclose(f);
       
       f = fopen("PartInner_Striped_BiCGStab_Res.txt", "w");
       fprintf(f, "Grid = {50,50} Diffusion 2d Stencil (0.001, pi/8)  %d x %d\n", A->global_num_rows, A->global_num_cols);
       for (int i=0; i<residuals_striped.size(); i++) {
           fprintf(f, "%lf\n", residuals_striped[i]);
       }
       fclose(f);
       
       f = fopen("PartInner_True_BiCGStab_Res.txt", "w");
       fprintf(f, "Grid = {50,50} Diffusion 2d Stencil (0.001, pi/8)  %d x %d\n", A->global_num_rows, A->global_num_cols);
       for (int i=0; i<residuals_true.size(); i++) {
           fprintf(f, "%lf\n", residuals_true[i]);
       }
       fclose(f);

    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) printf("Testing Contiguous Solution\n");
    for (int i = 0; i < x_true.local_n; i++) {
        assert(fabs(x_true.local[i] - x_contig.local[i]) < 1e-05);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) printf("Testing Striped Solution\n");
    for (int i = 0; i < x_true.local_n; i++) {
        assert(fabs(x_true.local[i] - x_striped.local[i]) < 1e-05);
    }

    MPI_Comm_free(&contig_comm);
    MPI_Comm_free(&striped_comm);

    delete[] stencil;
    delete A;

    MPI_Finalize();

    return 0;
}
