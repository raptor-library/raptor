#include <assert.h>

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    double b_val;
    int grid[2] = {25, 25};
    double eps = 0.001;
    double theta = M_PI/8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    CSRMatrix* A_sten = stencil_grid(stencil, grid, 2);

    Vector x(A_sten->n_rows);
    Vector b(A_sten->n_rows);
    
    // Test b <- A*ones
    x.set_const_value(1.0);
    A_sten->mult(x, b);
    FILE* f = fopen("../../tests/aniso_ones_b.txt", "r");
    for (int i = 0; i < A_sten->n_rows; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        assert(fabs(b[i] - b_val) < 1e-06);
    } 
    fclose(f);

    // Test b <- A_T*ones
    A_sten->mult_T(x, b);
    f = fopen("../../tests/aniso_ones_b_T.txt", "r");
    for (int i = 0; i < A_sten->n_rows; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        assert(fabs(b[i] - b_val) < 1e-06);
    } 
    fclose(f);

    // Tests b <- A*incr
    for (int i = 0; i < A_sten->n_rows; i++)
    {
        x[i] = i;
    }
    A_sten->mult(x, b);
    f = fopen("../../tests/aniso_inc_b.txt", "r");
    for (int i = 0; i < A_sten->n_rows; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        assert(fabs(b[i] - b_val) < 1e-06);
    } 
    fclose(f);

    // Tests b <- A_T*incr
    A_sten->mult_T(x, b);
    f = fopen("../../tests/aniso_inc_b_T.txt", "r");
    for (int i = 0; i < A_sten->n_rows; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        assert(fabs(b[i] - b_val) < 1e-06);
    } 
    fclose(f);



}
