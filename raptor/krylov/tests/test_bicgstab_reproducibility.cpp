#include <assert.h>
#include "raptor.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int grid[2] = {50, 50};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);

    ParVector x(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector b(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    std::vector<double> residuals, residuals2, residuals3, residuals4;

    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);
    BiCGStab(A, x, b, residuals);
   
    x.set_const_value(0.0);
    SeqInner_BiCGStab(A, x, b, residuals2);

    x.set_const_value(0.0);
    SeqNorm_BiCGStab(A, x, b, residuals3);

    x.set_const_value(0.0);
    SeqInnerSeqNorm_BiCGStab(A, x, b, residuals4);

    printf("PyAMG SeqIn SeqNorm SeqBoth\n");
    // Just testing the first 10 residuals
    if(rank == 0){
        FILE* f = fopen("../../../../test_data/bicgstab_res.txt", "r");
        double res;
        for (int i = 0; i < residuals.size(); i++)
        {
            fscanf(f, "%lf\n", &res);
	    //printf("%lf %lf %lf\n", res, residuals[i], fabs(res - residuals[i]));
	    printf("%lf %lf %lf %lf %lf\n", res, fabs(res-residuals[i]), fabs(res-residuals2[i]), fabs(res-residuals3[i]), fabs(res-residuals4[i]));
	    printf("%lf %lf %lf %lf %lf\n", res, residuals[i], residuals2[i], residuals3[i], residuals4[i]);
	    printf("----------------------------------------\n");
        }
        fclose(f);
    }
    MPI_Barrier(MPI_COMM_WORLD);


    delete[] stencil;
    delete A;

    MPI_Finalize();

    return 0;
}
