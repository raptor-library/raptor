#include <assert.h>
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "krylov/par_bicgstab.hpp"
#include "multilevel/par_multilevel.hpp"
#include "aggregation/par_smoothed_aggregation_solver.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_stencil.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int grid[2] = {50, 50};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    //double* stencil = diffusion_stencil_2d(0.1, M_PI/4.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    ParMultilevel *ml;

    ParVector x(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector b(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    aligned_vector<double> residuals;
    aligned_vector<double> pre_residuals;

    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);

    // BiCGStab
    BiCGStab(A, x, b, residuals);

    // Setup AMG hierarchy
    ml = new ParSmoothedAggregationSolver(0.0);
    ml->max_levels = 3;
    ml->setup(A);

    for (int i = 0; i < ml->num_levels; i++) {
        ParCSRMatrix* Al = ml->levels[i]->A;
        long lcl_nnz = Al->local_nnz;
        long nnz;
        MPI_Reduce(&lcl_nnz, &nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("%d\t%d\t%d\t%lu\n", i, Al->global_num_rows, Al->global_num_cols, nnz);
        }
    }
    
    for (int i = 0; i < ml->num_levels; i++) {
        ParCSRMatrix* Pl = ml->levels[i]->A;
        long lcl_nnz = Pl->local_nnz;
        long nnz;
        MPI_Reduce(&lcl_nnz, &nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("%d\t%d\t%d\t%lu\n", i, Pl->global_num_rows, Pl->global_num_cols, nnz);
        }
    }

    int iter = ml->solve(x, b);

    // AMG Preconditioned BiCGStab
    x.set_const_value(0.0);
    //Pre_BiCGStab(A, x, b, ml, pre_residuals);

    if (rank == 0) {
        FILE *f;
        f = fopen("BiCGStab_Res.txt", "w");
        fprintf(f, "2D Diffusion %d x %d\n", A->global_num_rows, A->global_num_cols);
        for (int i=0; i<residuals.size(); i++) {
            fprintf(f, "%lf \n", residuals[i]);
        }
        fclose(f);
        
        f = fopen("Pre_BiCGStab_Res.txt", "w");
        fprintf(f, "2D Diffusion %d x %d\n", A->global_num_rows, A->global_num_cols);
        for (int i=0; i<pre_residuals.size(); i++) {
            fprintf(f, "%lf \n", pre_residuals[i]);
        }
        fclose(f);
    }

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


    delete[] stencil;
    delete ml;
    delete A;

    MPI_Finalize();

    return 0;
}
