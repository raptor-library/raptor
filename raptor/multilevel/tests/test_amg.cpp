#include <assert.h>

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "multilevel/multilevel.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/external/mfem_wrapper.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    int dim;
    int n = 5;
    int system = 0;

    if (argc > 1)
    {
        system = atoi(argv[1]);
    }

    Multilevel* ml;
    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

    double strong_threshold = 0.25;

    int cache_len = 10000;
    double* cache_array = new double[cache_len];
    int num_tests = 10;

    if (system < 2)
    {
        double* stencil = NULL;
        std::vector<int> grid;
        if (argc > 2)
        {
            n = atoi(argv[2]);
        }

        if (system == 0)
        {
            dim = 3;
            grid.resize(dim, n);
            stencil = laplace_stencil_27pt();
        }
        else if (system == 1)
        {
            dim = 2;
            grid.resize(dim, n);
            double eps = 0.001;
            double theta = M_PI/8.0;
            strong_threshold = 0.0;
            if (argc > 3)
            {
                eps = atof(argv[3]);
                if (argc > 4)
                {
                    theta = atof(argv[4]);
                }
            }
            stencil = diffusion_stencil_2d(eps, theta);
        }
        A = par_stencil_grid(stencil, grid.data(), dim);
        delete[] stencil;
    }
/*    else if (system == 2)
    {
        char* mesh_file = "../../../../../mfem/data/beam-tet.mesh";
        int num_elements = 2;
        int order = 3;
        strong_threshold = 0.0;
        if (argc > 2)
        {
            num_elements = atoi(argv[2]);
            if (argc > 3)
            {
                order = atoi(argv[3]);
                if (argc > 4)
                {
                    mesh_file = argv[4];
                }
            }
        }
        A = mfem_linear_elasticity(mesh_file, num_elements, order);
    }
*/    else if (system == 3)
    {
        char* file = "../../../../examples/LFAT5.mtx";
        A = readParMatrix(file, MPI_COMM_WORLD, 1, 1);
        if (argc > 2)
        {
            strong_threshold = atof(argv[2]);
        }
    }

    x.resize(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    b.resize(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);
    
    ml = new Multilevel(A, strong_threshold);

    if (rank == 0)
    {
        printf("Num Levels = %d\n", ml->num_levels);
	printf("A\tNRow\tNCol\tNNZ\n");
    }
    for (int i = 0; i < ml->num_levels; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
	long lcl_nnz = Al->local_nnz;
	long nnz;
	MPI_Reduce(&lcl_nnz, &nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	if (rank == 0)
	{
            printf("%d\t%d\t%d\t%lu\n", i, Al->global_num_rows, Al->global_num_cols, nnz);
        }
    }

    if (rank == 0)
    {
	printf("\nP\tNRow\tNCol\tNNZ\n");
    }
    for (int i = 0; i < ml->num_levels-1; i++)
    {
        ParCSRMatrix* Pl = ml->levels[i]->P;
	long lcl_nnz = Pl->local_nnz;
	long nnz;
	MPI_Reduce(&lcl_nnz, &nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	if (rank == 0)
	{
            printf("%d\t%d\t%d\t%lu\n", i, Pl->global_num_rows, Pl->global_num_cols, nnz);
	}
    }
    
    if (rank == 0)
    {
        printf("\nSolve Phase Relative Residuals:\n");
    }
    ml->solve(x, b);

    delete ml;
    delete A;

    MPI_Finalize();

    return 0;
}



