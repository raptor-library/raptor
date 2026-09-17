// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_cg.hpp"
#include "raptor/multilevel/par_multilevel.hpp"

namespace raptor {

void CG(ParCSRMatrix* A, ParVector& x, ParVector& b, std::vector<double>& res, double tol, int max_iter, double* comm_t)
{
    int rank;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);

    ParVector r;
    ParVector p;
    ParVector Ap;

    int iter, recompute_r;
    data_t alpha, beta;
    data_t rr_inner, next_inner, App_inner;
    double norm_r;
    double b_norm = b.norm(2);
    if (b_norm < zero_tol) b_norm = 1.0;

    if (max_iter <= 0)
    {
        max_iter = ((int)(1.3*b.global_n)) + 2;
    }

    // Fixed Constructors
    r.resize(b.global_n, b.local_n);
    p.resize(b.global_n, b.local_n);
    Ap.resize(b.global_n, b.local_n);

    // r0 = b - A * x0
    A->residual(x, b, r);

    // p0 = r0
    p.copy(r);

if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
    rr_inner = r.inner_product(r);
if (comm_t) *comm_t += RAPtor_MPI_Wtime();
    norm_r = sqrt(rr_inner);
    res.emplace_back(norm_r / b_norm);

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
if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
        App_inner = Ap.inner_product(p);
if (comm_t) *comm_t += RAPtor_MPI_Wtime();
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
if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
        next_inner = r.inner_product(r);
if (comm_t) *comm_t += RAPtor_MPI_Wtime();
        beta = next_inner / rr_inner;

        // p_{i+1} = r_{i+1} + beta_i * p_i
        p.scale(beta);
        p.axpy(r, 1.0);

        // Update next inner product
        rr_inner = next_inner;
        norm_r = sqrt(rr_inner);
        res.emplace_back(norm_r / b_norm);

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

template <class T, is_bsr_or_csr<T> = true>
void PCG_helper(T* A, ParMultilevel_T<T>* ml, ParVector& x, ParVector& b, std::vector<double>& res, double tol, int max_iter, double* precond_t, double* comm_t)
{
    int rank;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);

    ParVector r;
    ParVector z;
    ParVector p;
    ParVector Ap;

    int iter;
    int recompute_r = 4;
    bool full_r;
    data_t alpha, beta;
    data_t rz_inner = 0.0;
    data_t next_inner, App_inner;
    double b_norm;
    double rel_res;
if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
    b_norm = b.norm(2);
if (comm_t) *comm_t += RAPtor_MPI_Wtime();
    if (b_norm < zero_tol) b_norm = 1.0;

    if (max_iter <= 0)
    {
        max_iter = ((int)(1.3*b.global_n)) + 2;
    }

    // Fixed Constructors
    r.resize(b.global_n, b.local_n);
    z.resize(b.global_n, b.local_n);
    p.resize(b.global_n, b.local_n);
    Ap.resize(b.global_n, b.local_n);

    // r0 = b - A * x0
    A->residual(x, b, r);
if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
    rel_res = r.norm(2) / b_norm;
if (comm_t) *comm_t += RAPtor_MPI_Wtime();
    res.emplace_back(rel_res);

    if (rel_res > tol)
    {
        // z = M^{-1}r0
        z.set_const_value(0.0);
if (precond_t) *precond_t -= RAPtor_MPI_Wtime();
        ml->cycle(z, r);
if (precond_t) *precond_t += RAPtor_MPI_Wtime();

        // p0 = z0
        p.copy(z);

        // <r, z>
if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
        rz_inner = r.inner_product(z);
if (comm_t) *comm_t += RAPtor_MPI_Wtime();
    }

    recompute_r = 8;
    iter = 0;

    // Main CG Loop
    while (rel_res > tol && iter < max_iter)
    {
        iter++;

        // alpha_i = (r_i, z_i) / (A*p_i, p_i)
        A->mult(p, Ap);
if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
        App_inner = Ap.inner_product(p);
if (comm_t) *comm_t += RAPtor_MPI_Wtime();
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

        full_r = recompute_r && iter % recompute_r == 0;

        if (full_r)
        {
            A->residual(x, b, r);
        }
        else
        {
            r.axpy(Ap, -1.0*alpha);
        }

        // stop on ||b - A*x||_2 / ||b||_2.
if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
        rel_res = r.norm(2) / b_norm;
if (comm_t) *comm_t += RAPtor_MPI_Wtime();
        res.emplace_back(rel_res);
        if (rel_res <= tol) break;

        // z_{j+1} = M^{-1}r_{j+1}
        z.set_const_value(0.0);
if (precond_t) *precond_t -= RAPtor_MPI_Wtime();
        ml->cycle(z, r);
if (precond_t) *precond_t += RAPtor_MPI_Wtime();

        // beta_i = (r_{i+1}, z_{i+1}) / (r_i, z_i)
if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
        next_inner = r.inner_product(z);
if (comm_t) *comm_t += RAPtor_MPI_Wtime();
        beta = next_inner / rz_inner;

        // p_{i+1} = z_{i+1} + beta_i * p_i
        p.scale(beta);
        p.axpy(z, 1.0);

        // Update next inner product
        rz_inner = next_inner;
    }

    if (rank == 0)
    {
        if (iter == max_iter && rel_res > tol)
        {
            printf("Max Iterations Reached.\n");
        }
        else
        {
            printf("%d Iteration required to converge\n", iter);
        }
        printf("Relative Residual: %lg\n\n", res.back());
    }

    return;
}

template <class T, is_bsr_or_csr<T>>
void PCG(T* A, ParMultilevel_T<T>* ml, ParVector& x, ParVector& b,
        std::vector<double>& res, double tol, int max_iter,
        double* precond_t, double* comm_t)
{
        PCG_helper(A, ml, x, b, res, tol, max_iter, precond_t, comm_t);
}

template void PCG(ParCSRMatrix* A, ParMultilevel* ml, ParVector& x, ParVector& b,
        std::vector<double>& res, double tol, int max_iter,
        double* precond_t, double* comm_t);
template void PCG(ParBSRMatrix* A, ParMultilevel_T<ParBSRMatrix>* ml, ParVector& x, ParVector& b,
        std::vector<double>& res, double tol, int max_iter,
        double* precond_t, double* comm_t);
}
