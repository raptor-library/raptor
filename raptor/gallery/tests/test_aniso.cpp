#include <assert.h>

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"
#include "gallery/matrix_IO.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    int n_rows, n_cols; 
    int row, row_nnz, nnz;
    int start, end;
    double row_sum, sum;
    int grid[2] = {25, 25};
    double eps = 0.001;
    double theta = M_PI/8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    CSRMatrix* A_sten = stencil_grid(stencil, grid, 2);
    CSRMatrix* A_io = readMatrix("../../tests/aniso.mtx", 1);
     
    // Open laplacian data file
    FILE *f = fopen("../../tests/aniso_data.txt", "r");

    // Read global shape
    fscanf(f, "%d %d\n", &n_rows, &n_cols);

    // Compare shapes
    assert(n_rows == A_sten->n_rows);
    assert(n_cols == A_sten->n_cols);
    assert(n_rows == A_io->n_rows);
    assert(n_cols == A_io->n_cols);

    A_sten->sort();
    A_sten->remove_duplicates();

    A_io->sort();
    A_io->remove_duplicates();

    assert(A_sten->idx1[0] == A_io->idx1[0]);
    for (int i = 0; i < n_rows; i++)
    {
        fscanf(f, "%d %d %lg\n", &row, &row_nnz, &row_sum);

        // Check correct row_ptrs
        assert(A_sten->idx1[i+1] == A_io->idx1[i+1]);
        start = A_sten->idx1[i];
        end = A_sten->idx1[i+1];
        assert(end - start == row_nnz);

        // Check correct col indices / values
        sum = 0;
        for (int j = start; j < end; j++)
        {
            sum += A_sten->vals[j];
            assert(A_sten->idx2[j] == A_io->idx2[j]);
            assert(fabs(A_sten->vals[j] - A_io->vals[j]) < 1e-12);
        }
        assert(fabs(row_sum - sum) < 1e-05);
    }

    fclose(f);

    delete A_io;
    delete[] stencil;
    delete A_sten;
}
