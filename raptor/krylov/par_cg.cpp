// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "krylov/par_cg.hpp"
#include "multilevel/par_multilevel.hpp"

using namespace raptor;

void CG(ParCSRMatrix* A, ParVector& x, ParVector& b, std::vector<double>& res, double tol, int max_iter)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ParVector r;
    ParVector p;
    ParVector Ap;

    int iter, recompute_r;
    data_t alpha, beta;
    data_t rr_inner, next_inner, App_inner;
    double norm_r;

    if (max_iter <= 0)
    {
        max_iter = ((int)(1.3*b.global_n)) + 2;
    }

    // Fixed Constructors
    r.resize(b.global_n, b.local_n, b.first_local);
    p.resize(b.global_n, b.local_n, b.first_local);
    Ap.resize(b.global_n, b.local_n, b.first_local);

    // r0 = b - A * x0
    A->residual(x, b, r);

    // p0 = r0
    p.copy(r);

    rr_inner = r.inner_product(r);
    norm_r = sqrt(rr_inner);
    res.push_back(norm_r);

    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    // How often should r be recomputed
    recompute_r = 8;
    iter = 0;

    // Main CG Loop
    while (norm_r > tol && iter < max_iter)
    {
        // alpha_i = (r_i, r_i) / (A*p_i, p_i)
        A->mult(p, Ap);
        App_inner = Ap.inner_product(p);
        if (App_inner < 0.0)
        {
            if (rank == 0)
            {
                printf("Indefinite matrix detected in CG! Aborting...\n");
            }
            exit(-1);
        }
        alpha = rr_inner / App_inner;

        x.axpy(p, alpha);

        // x_{i+1} = x_i + alpha_i * p_i
        if ((iter % recompute_r) && iter > 0)
        {
            r.axpy(Ap, -1.0*alpha);
        }
        else
        {
            A->residual(x, b, r);
        }

        // beta_i = (r_{i+1}, r_{i+1}) / (r_i, r_i)
        next_inner = r.inner_product(r);
        beta = next_inner / rr_inner;

        // p_{i+1} = r_{i+1} + beta_i * p_i
        p.scale(beta);
        p.axpy(r, 1.0);

        // Update next inner product
        rr_inner = next_inner;
        norm_r = sqrt(rr_inner);

        iter++;
    }

    if (rank == 0)
    {
        if (iter == max_iter)
        {
            printf("Max Iterations Reached.\n");
            printf("2 Norm of Residual: %lg\n\n", norm_r);
        }
        else
        {
            printf("%d Iteration required to converge\n", iter);
            printf("2 Norm of Residual: %lg\n\n", norm_r);
        }
    }

    return;
}


void PCG(ParCSRMatrix* A, ParMultilevel* ml, ParVector& x, ParVector& b, std::vector<double>& res, double tol, int max_iter)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ParVector r;
    ParVector z;
    ParVector p;
    ParVector Ap;

    int iter, recompute_r;
    data_t alpha, beta;
    data_t rz_inner, next_inner, App_inner;
    double norm_rz;

    if (max_iter <= 0)
    {
        max_iter = ((int)(1.3*b.global_n)) + 2;
    }

    // Fixed Constructors
    r.resize(b.global_n, b.local_n, b.first_local);
    z.resize(b.global_n, b.local_n, b.first_local);
    p.resize(b.global_n, b.local_n, b.first_local);
    Ap.resize(b.global_n, b.local_n, b.first_local);

    // r0 = b - A * x0
    A->residual(x, b, r);

    // z0 = M^{-1}r0
    z.set_const_value(0.0);
    ml->solve(z, r, NULL, 1); // preconditioner = single iteration of AMG

    // p0 = z0
    p.copy(z);

    rz_inner = r.inner_product(z);
    norm_rz = sqrt(rz_inner);
    res.push_back(norm_rz);  // Residuals are preconditioned residuals...

    if (norm_rz != 0.0)
    {
        tol = tol * norm_rz;
    }

    // How often should r be recomputed
    recompute_r = 8;
    iter = 0;

    // Main CG Loop
    while (norm_rz > tol && iter < max_iter)
    {
        // alpha_i = (r_i, r_i) / (A*p_i, p_i)
        A->mult(p, Ap);
        App_inner = Ap.inner_product(p);
        if (App_inner < 0.0)
        {
            if (rank == 0)
            {
                printf("Indefinite matrix detected in CG! Aborting...\n");
            }
            exit(-1);
        }
        alpha = rz_inner / App_inner;

        // x_{i+1} = x_i + alpha_i * p_i
        x.axpy(p, alpha);

        if ((iter % recompute_r) && iter > 0)
        {
            r.axpy(Ap, -1.0*alpha);
        }
        else
        {
            A->residual(x, b, r);
        }

        // z_{j+1} = M^{-1}r_{j+1}
        z.set_const_value(0.0);
        ml->solve(z, r, NULL, 1); // Single iteration of AMG for z_{j+1}

        // beta_i = (r_{i+1}, z_{i+1}) / (r_i, z_i)
        next_inner = r.inner_product(z);
        beta = next_inner / rz_inner;

        // p_{i+1} = r_{i+1} + beta_i * p_i
        p.scale(beta);
        p.axpy(z, 1.0);

        // Update next inner product
        rz_inner = next_inner;
        norm_rz = sqrt(rz_inner);

        iter++;
    }

    if (rank == 0)
    {
        if (iter == max_iter)
        {
            printf("Max Iterations Reached.\n");
            printf("2 Norm of Residual: %lg\n\n", norm_rz);
        }
        else
        {
            printf("%d Iteration required to converge\n", iter);
            printf("2 Norm of Residual: %lg\n\n", norm_rz);
        }
    }

    return;
}


void tap_PCG(ParCSRMatrix* A, ParMultilevel* ml, ParVector& x, ParVector& b, std::vector<double>& res, double tol, int max_iter)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ParVector r;
    ParVector z;
    ParVector p;
    ParVector Ap;

    int iter, recompute_r;
    data_t alpha, beta;
    data_t rz_inner, next_inner, App_inner;
    double norm_rz;

    if (max_iter <= 0)
    {
        max_iter = ((int)(1.3*b.global_n)) + 2;
    }

    // Fixed Constructors
    r.resize(b.global_n, b.local_n, b.first_local);
    z.resize(b.global_n, b.local_n, b.first_local);
    p.resize(b.global_n, b.local_n, b.first_local);
    Ap.resize(b.global_n, b.local_n, b.first_local);

    // r0 = b - A * x0
    A->residual(x, b, r);

    // z0 = M^{-1}r0
    z.set_const_value(0.0);
    ml->tap_solve(z, r, NULL, 1); // preconditioner = single iteration of AMG

    // p0 = z0
    p.copy(z);

    rz_inner = r.inner_product(z);
    norm_rz = sqrt(rz_inner);
    res.push_back(norm_rz);  // Residuals are preconditioned residuals...

    if (norm_rz != 0.0)
    {
        tol = tol * norm_rz;
    }

    // How often should r be recomputed
    recompute_r = 8;
    iter = 0;

    // Main CG Loop
    while (norm_rz > tol && iter < max_iter)
    {
        // alpha_i = (r_i, r_i) / (A*p_i, p_i)
        A->mult(p, Ap);
        App_inner = Ap.inner_product(p);
        if (App_inner < 0.0)
        {
            if (rank == 0)
            {
                printf("Indefinite matrix detected in CG! Aborting...\n");
            }
            exit(-1);
        }
        alpha = rz_inner / App_inner;

        // x_{i+1} = x_i + alpha_i * p_i
        x.axpy(p, alpha);

        if ((iter % recompute_r) && iter > 0)
        {
            r.axpy(Ap, -1.0*alpha);
        }
        else
        {
            A->residual(x, b, r);
        }

        // z_{j+1} = M^{-1}r_{j+1}
        z.set_const_value(0.0);
        ml->tap_solve(z, r, NULL, 1); // Single iteration of AMG for z_{j+1}

        // beta_i = (r_{i+1}, z_{i+1}) / (r_i, z_i)
        next_inner = r.inner_product(z);
        beta = next_inner / rz_inner;

        // p_{i+1} = r_{i+1} + beta_i * p_i
        p.scale(beta);
        p.axpy(z, 1.0);

        // Update next inner product
        rz_inner = next_inner;
        norm_rz = sqrt(rz_inner);

        iter++;
    }

    if (rank == 0)
    {
        if (iter == max_iter)
        {
            printf("Max Iterations Reached.\n");
            printf("2 Norm of Residual: %lg\n\n", norm_rz);
        }
        else
        {
            printf("%d Iteration required to converge\n", iter);
            printf("2 Norm of Residual: %lg\n\n", norm_rz);
        }
    }

    return;
}


