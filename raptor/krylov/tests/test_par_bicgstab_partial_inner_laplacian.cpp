#include <assert.h>
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "krylov/par_bicgstab.hpp"
#include "gallery/laplacian27pt.hpp"
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
    int grid[3] = {10, 10, 10};
    double* stencil = laplace_stencil_27pt();
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 3);
    
    ParVector x(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
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
    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);
    PI_BiCGStab(A, x, b, residuals_contig, contig_comm, color, first_root, second_root, part_global, 0);

    MPI_Barrier(MPI_COMM_WORLD);

    // Test striped even procs
    //create_partial_inner_comm(striped_comm, color, first_root, second_root, part_global, 1);
    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);
    PI_BiCGStab(A, x, b, residuals_striped, striped_comm, color, first_root, second_root, part_global, 1);

    MPI_Barrier(MPI_COMM_WORLD);

    // Write out residuals to file
    if (rank == 0) {
       FILE *f;
       f = fopen("PartInner_Contig_BiCGStab_Res.txt", "w");
       fprintf(f, "Grid = {10, 10, 10} Laplacian 3d 27-pt Stencil %d x %d\n", A->global_num_rows, A->global_num_cols);
       for (int i=0; i<residuals_contig.size(); i++) {
           fprintf(f, "%lf \n", residuals_contig[i]);
       }
       fclose(f);
       
       f = fopen("PartInner_Striped_BiCGStab_Res.txt", "w");
       fprintf(f, "Grid = {10, 10, 10} Laplacian 3d 27-pt Stencil %d x %d\n", A->global_num_rows, A->global_num_cols);
       for (int i=0; i<residuals_striped.size(); i++) {
           fprintf(f, "%lf\n", residuals_striped[i]);
       }
       fclose(f);
       
       f = fopen("PartInner_True_BiCGStab_Res.txt", "w");
       fprintf(f, "Grid = {10, 10, 10} Laplacian 3d 27-pt Stencil %d x %d\n", A->global_num_rows, A->global_num_cols);
       for (int i=0; i<residuals_true.size(); i++) {
           fprintf(f, "%lf\n", residuals_true[i]);
       }
       fclose(f);

    }

    // Rethink check for convergence
    /*if(rank == 0){
        FILE* f = fopen("../../../../test_data/bicgstab_res.txt", "r");
        //FILE* f = fopen("../../../../test_data/bicgstab_res_TEST.txt", "r");
        double res;
        for (int i = 0; i < 20; i++)
        {
            fscanf(f, "%lf\n", &res);
	    //printf("%lf %lf %lf\n", res, residuals[i], fabs(res - residuals[i]));
            assert(fabs(res - residuals[i]) < 1e-06);
        }
        fclose(f);
    }
    MPI_Barrier(MPI_COMM_WORLD);*/

    MPI_Comm_free(&contig_comm);
    MPI_Comm_free(&striped_comm);

    delete[] stencil;
    delete A;

    MPI_Finalize();

    return 0;
}
