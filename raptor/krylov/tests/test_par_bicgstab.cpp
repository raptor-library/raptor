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

    int grid[2] = {50, 50};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);

    //printf("A size: %d %d\n", A->global_num_rows, A->global_num_rows);
    ParVector x(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector b(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    std::vector<double> residuals;

    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);

    /*for(int i=0; i<num_procs; i++){
        if(i == rank){
	    b.local.print();
	}
        MPI_Barrier(MPI_COMM_WORLD);
    }*/

    data_t temp;
    temp = b.inner_product(b);

    printf("%lf\n", temp);

    BiCGStab(A, x, b, residuals);

    /*for(int i=0; i<num_procs; i++){
        if(i == rank){
	    x.local.print();
	}
        MPI_Barrier(MPI_COMM_WORLD);
    }*/

    // CHANGE ALL OF THIS FOR BICGSTAB
    if(rank == 0){
    FILE* f = fopen("../../../../test_data/bicgstab_res_v2.txt", "r");
    double res;
    printf("Pyamg -- RAPtor- Difference\n");
    for (int i = 0; i < residuals.size(); i++)
    {
        fscanf(f, "%lf\n", &res);
	printf("%lf %lf %lf\n", res, residuals[i], fabs(res - residuals[i]));
        //assert(fabs(res - residuals[i]) < 1e-06);
    }
    fclose(f);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Done\n");


    delete[] stencil;
    delete A;

    MPI_Finalize();

    return 0;
}
