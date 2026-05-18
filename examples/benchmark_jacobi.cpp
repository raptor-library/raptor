// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include "clear_cache.hpp"
#include <ctime>

#include "raptor.hpp"
#include "raptor-sparse.hpp"

using namespace raptor;

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
    Vector tmp;
    Vector r;
    Vector x_jac;

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
    x_jac = Vector(A->n_rows);
    b = Vector(A->n_rows);
    tmp = Vector(A->n_rows);
    r = Vector(A->n_rows);

    b.set_rand_values();
    x.set_const_value(0.0);

    int n_outer = 10;
    int n_inner = 2;
    double r_norm, b_norm;
    b_norm = b.norm(2);

    printf("Standard Jacobi:\n");
    for (int i = 0; i < A->n_rows; i++)
        x_jac[i] = x[i];
    A->residual(x_jac, b, r);
    r_norm = r.norm(2);
    printf("Initial: %e\n", r_norm/b_norm);
    for (int i = 0; i < n_outer; i++)
    {
        sor(A, b, x_jac, tmp, n_inner, 2.0/3, NULL, NULL, 0, 1.0);
        A->residual(x_jac, b, r);
        r_norm = r.norm(2);
        printf("%d: %e\n", i, r_norm/b_norm); 
    }

    printf("Sparsified Jacobi, S 1.005:\n");
    for (int i = 0; i < A->n_rows; i++)
        x_jac[i] = x[i];
    A->residual(x_jac, b, r);
    r_norm = r.norm(2);
    printf("Initial: %e\n", r_norm/b_norm);
    for (int i = 0; i < n_outer; i++)
    {
        sor(A, b, x_jac, tmp, n_inner, 2.0/3, NULL, NULL, 0, 1.005);
        A->residual(x_jac, b, r);
        r_norm = r.norm(2);
        printf("%d: %e\n", i, r_norm/b_norm); 
    }

    printf("Sparsified Jacobi, S 1.01:\n");
    for (int i = 0; i < A->n_rows; i++)
        x_jac[i] = x[i];
    A->residual(x_jac, b, r);
    r_norm = r.norm(2);
    printf("Initial: %e\n", r_norm/b_norm);
    for (int i = 0; i < n_outer; i++)
    {
        sor(A, b, x_jac, tmp, n_inner, 2.0/3, NULL, NULL, 0, 1.01);
        A->residual(x_jac, b, r);
        r_norm = r.norm(2);
        printf("%d: %e\n", i, r_norm/b_norm); 
    }

    printf("Sparsified Jacobi, S 1.05:\n");
    for (int i = 0; i < A->n_rows; i++)
        x_jac[i] = x[i];
    A->residual(x_jac, b, r);
    r_norm = r.norm(2);
    printf("Initial: %e\n", r_norm/b_norm);
    for (int i = 0; i < n_outer; i++)
    {
        sor(A, b, x_jac, tmp, n_inner, 2.0/3, NULL, NULL, 0, 1.05);
        A->residual(x_jac, b, r);
        r_norm = r.norm(2);
        printf("%d: %e\n", i, r_norm/b_norm); 
    }

    printf("Sparsified Jacobi, S 1.1:\n");
    for (int i = 0; i < A->n_rows; i++)
        x_jac[i] = x[i];
    A->residual(x_jac, b, r);
    r_norm = r.norm(2);
    printf("Initial: %e\n", r_norm/b_norm);
    for (int i = 0; i < n_outer; i++)
    {
        sor(A, b, x_jac, tmp, n_inner, 2.0/3, NULL, NULL, 0, 1.1);
        A->residual(x_jac, b, r);
        r_norm = r.norm(2);
        printf("%d: %e\n", i, r_norm/b_norm); 
    }

    printf("Sparsified Jacobi, S 1.5:\n");
    for (int i = 0; i < A->n_rows; i++)
        x_jac[i] = x[i];
    A->residual(x_jac, b, r);
    r_norm = r.norm(2);
    printf("Initial: %e\n", r_norm/b_norm);
    for (int i = 0; i < n_outer; i++)
    {
        sor(A, b, x_jac, tmp, n_inner, 2.0/3, NULL, NULL, 0, 1.5);
        A->residual(x_jac, b, r);
        r_norm = r.norm(2);
        printf("%d: %e\n", i, r_norm/b_norm); 
    }

    printf("Sparsified Jacobi, S 2:\n");
    for (int i = 0; i < A->n_rows; i++)
        x_jac[i] = x[i];
    A->residual(x_jac, b, r);
    r_norm = r.norm(2);
    printf("Initial: %e\n", r_norm/b_norm);
    for (int i = 0; i < n_outer; i++)
    {
        sor(A, b, x_jac, tmp, n_inner, 2.0/3, NULL, NULL, 0, 2);
        A->residual(x_jac, b, r);
        r_norm = r.norm(2);
        printf("%d: %e\n", i, r_norm/b_norm); 
    }

    printf("Sparsified Jacobi, Dynamic:\n");
    float S = 1.5;
    for (int i = 0; i < A->n_rows; i++)
        x_jac[i] = x[i];
    A->residual(x_jac, b, r);
    r_norm = r.norm(2);
    printf("Initial: %e\n", r_norm/b_norm);
    for (int i = 0; i < n_outer; i++)
    {
        sor(A, b, x_jac, tmp, n_inner, 2.0/3, NULL, NULL, 0, S);
        A->residual(x_jac, b, r);
        r_norm = r.norm(2);
        printf("%d, S %f: %e\n", i, S, r_norm/b_norm); 
        S = 1.0 + (S - 1.0)/2;
    }

    delete A;

    return 0;
}


