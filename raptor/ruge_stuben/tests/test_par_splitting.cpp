#include <assert.h>

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "ruge_stuben/par_cf_splitting.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_stencil.hpp"
#include <iostream>
#include <fstream>

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FILE* f;
    ParCSRMatrix* S;
    std::vector<int> splitting;
    std::vector<int> splitting_rap;
    std::vector<int> off_proc_states;

    /* TEST 10 x 10 2D rotated aniso... print this one for graphing */
    int grid[2] = {10, 10};
    double eps = 0.001;
    double theta = M_PI/8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    S = A->strength();
    split_falgout(S, splitting_rap, off_proc_states);

    char name[128];
    snprintf(name, sizeof(name), "aniso_splitting_%d_%d.txt", num_procs, rank);    
    ofstream outfile;
    outfile.open(name);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        outfile << splitting_rap[i] << endl;
    }
    outfile.close();

    MPI_Finalize();

    return 0;
}
