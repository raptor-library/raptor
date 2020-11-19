// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include "clear_cache.hpp"
#include <ctime>

#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "core/types.hpp"
#include "gallery/stencil.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/matrix_IO.hpp"

//using namespace raptor;

double wtime()
{
    return (double) clock() / (double) CLOCKS_PER_SEC;
}

int main(int argc, char *argv[])
{
    int dim=0;
    int n = 5;
    int system = 0;

    if (argc > 1)
    {
        system = atoi(argv[1]);
    }

    CSRMatrix* A=NULL;
    Vector x;
    Vector b;

    double t0, tfinal;

    int num_tests = 2;

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
        A = stencil_grid(stencil, grid.data(), dim);
        delete[] stencil;
    }
    else if (system == 3)
    {
        const char* file = "../../test_data/rss_A0.pm";

        if (argc > 2)
        {
            file = argv[2];
        }
        A = readMatrix(file);
    }

    x = Vector(A->n_rows);
    b = Vector(A->n_rows);

    x.set_const_value(1.0);

    int n_spmvs = 10000;

    int* idx1 = new int[A->idx1.size()];
    int* idx2 = new int[A->idx2.size()];
    double* vals = new double[A->vals.size()];
    double* b_ptr = new double[A->n_rows];
    double* x_ptr = new double[A->n_rows];
    idx1[0] = 0;
    for (int i = 0; i < A->n_rows; i++)
    {
        idx1[i+1] = A->idx1[i+1];
        for (int j = idx1[i]; j < idx1[i+1]; j++)
        {
            idx2[j] = A->idx2[j];
            vals[j] = A->vals[j];
        }
        x_ptr[i] = 1.0;
    }

    int start, end;
    for (int i = 0; i < num_tests; i++)
    {
        t0 = wtime();
        for (int j = 0; j < n_spmvs; j++)
        {
            A->mult(x, b);
        }
        tfinal = (wtime() - t0) / n_spmvs;
        printf("Standard SpMV Time: %e\n", tfinal);

        t0 = wtime();
        for (int j = 0; j < n_spmvs; j++)
        {
            for (int k = 0; k < A->n_rows; k++)
            {
                b_ptr[k] = 0.0;
            }
            start = 0;
            for (int k = 0; k < A->n_rows; k++)
            {
                end = idx1[k+1];
                for (int l = start; l < end; l++)
                {
                    b_ptr[k] += vals[l] * x_ptr[idx2[l]];
                }
                start = end;
            }
        }
        tfinal = (wtime() - t0) / n_spmvs;
        printf("Pointer SpMV Time: %e\n", tfinal);
    }

    delete A;

    return 0;
}

