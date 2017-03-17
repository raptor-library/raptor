#include <assert.h>
#include <math.h>
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/stencil.hpp"
#include "gallery/diffusion.hpp"
#include "multilevel/multilevel.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    double eps = 0.001;
    double theta = M_PI/8.0;
    int grid[2] = {10, 10};
    double* stencil = diffusion_stencil_2d(eps, theta);

    CSRMatrix A;
    stencil_grid(&A, stencil, grid, 2);
    A.sort();

    Multilevel ml(A, 0, 4.0/3, 1, 50);

    CSCMatrix& P = ml.levels[0]->P;
    
    CSRMatrix Ac;
    CSRMatrix Atmp;
    A.mult(P, &Atmp);
    P.mult_T(Atmp, &Ac);

    CSRMatrix Ac_RAP;
    A.RAP(P, &Ac_RAP);
    
    Ac.sort();
    Ac_RAP.sort();

    // Make sure more than 0 rows, cols, nonzeros
    assert(Ac.n_rows > 0);
    assert(Ac.n_cols > 0);
    assert(Ac.nnz > 0);

    // Assert equal dimensions between PT*(A*P) and RAP
    assert(Ac.n_rows == Ac_RAP.n_rows);
    assert(Ac.n_cols == Ac_RAP.n_cols);
    assert(Ac.nnz == Ac_RAP.nnz);

    for (int i = 0; i < Ac.n_rows; i++)
    {
        int row_start = Ac.idx1[i];
        int row_end = Ac.idx1[i+1];
        assert(row_start == Ac_RAP.idx1[i]);
        assert(row_end == Ac_RAP.idx1[i+1]);
        for (int j = row_start; j < row_end; j++)
        {
            assert(Ac.idx2[j] == Ac_RAP.idx2[j]);
            assert(fabs((Ac.vals[j] - Ac_RAP.vals[j])/Ac.vals[j]) < 1e-06);
        }
    }

    delete[] stencil;

}

