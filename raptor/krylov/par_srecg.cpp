// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "krylov/par_srecg.hpp"
#include "multilevel/par_multilevel.hpp"
#include "aorthonormalization/par_cgs.hpp"

using namespace raptor;

void SRECG(ParCSRMatrix* A, ParVector& x, ParVector& b, int t, aligned_vector<double>& res, 
        double tol, int max_iter, double* comm_t)
{
    int rank;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);

    ParVector r;
    ParBVector *W = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *W_temp = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *Wk_1 = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *Wk_2 = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParVector Wa;
    ParVector T;
    Vector alpha;

    int iter, recompute_r;
    data_t rr_inner;
    double norm_r;

    // Adjust max iterations
    if (max_iter <= 0)
    {
        max_iter = ((int)(1.3*b.global_n)) + 2;
    }

    // Fixed Constructors
    r.resize(b.global_n, b.local_n);
    Wa.resize(b.global_n, b.local_n);
    T.resize(b.global_n, b.local_n);
    alpha.resize(t);

    // r0 = b - A * x0
    A->residual(x, b, r);

    // Append norm of initial residual to res
if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
    rr_inner = r.inner_product(r);
if (comm_t) *comm_t += RAPtor_MPI_Wtime();
    norm_r = sqrt(rr_inner);
    res.emplace_back(norm_r);

    // Adjust tolerance
    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    // Perform first iteration outside loop to reduce
    // control flow instructions
    iter = 0;

    // W = T(r0)
    r.split_contig(*W, t, A->partition->first_local_row);

    // A-orthonormalize W
    CGS(A, *W);

    // alpha = W^T * r
    W->mult_T(r, alpha);

    // x = x + W * alpha
    W->mult(alpha, Wa);
    x.axpy(Wa, 1.0);

    // r = r - A * W * alpha
    A->mult(Wa, T);
    r.axpy(T, -1.0);

    // Update norm of residual and iteration
if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
    rr_inner = r.inner_product(r);
if (comm_t) *comm_t += RAPtor_MPI_Wtime();
    norm_r = sqrt(rr_inner);
    res.emplace_back(norm_r);
    iter++;

    // Main SRECG Loop
    while (norm_r > tol && iter < max_iter)
    {
        if (iter > 1)
        {
            // Update Wk_1 and Wk_2
            Wk_2->copy(*Wk_1);
            Wk_1->copy(*W);

            // W = A * W
            A->mult(*W, *W_temp);
            BCGS(A, *Wk_1, *Wk_2, *W_temp);
        }
        else
        {
            // Update Wk_1
            Wk_1->copy(*W);

            // W = A * W
            A->mult(*W, *W_temp);
            BCGS(A, *Wk_1, *W_temp);
        }

        W->copy(*W_temp);

        // A-orthonormalize W
        CGS(A, *W);

        // alpha = W^T * r
        W->mult_T(r, alpha);

        // x = x + W * alpha
        W->mult(alpha, Wa);
        x.axpy(Wa, 1.0);

        // r = r - A * W * alpha
        A->mult(Wa, T);
        r.axpy(T, -1.0);

        // Update norm of residual and iteration
    if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
        rr_inner = r.inner_product(r);
    if (comm_t) *comm_t += RAPtor_MPI_Wtime();
        norm_r = sqrt(rr_inner);
        res.emplace_back(norm_r);

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

    delete W;
    delete Wk_1;
    delete Wk_2;
    delete W_temp;

    return;
}

void PSRECG(ParCSRMatrix* A, ParMultilevel* ml_single, ParMultilevel* ml, ParVector& x, ParVector& b,
    int t, aligned_vector<double>& res, double tol, int max_iter, double* precond_t, double* comm_t)
{
    int rank;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    
    ParVector r;
    ParVector z;
    ParBVector *W = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *W_temp = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *Wk_1 = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *Wk_2 = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParVector Wa;
    ParVector T;
    Vector alpha;
    
    int iter, recompute_r;
    data_t rr_inner;
    double norm_r;

    // Adjust max iterations
    if (max_iter <= 0)
    {
        max_iter = ((int)(1.3*b.global_n)) + 2;
    }

    // Fixed Constructors
    r.resize(b.global_n, b.local_n);
    z.resize(b.global_n, b.local_n);
    Wa.resize(x.global_n, x.local_n);
    T.resize(x.global_n, x.local_n);
    alpha.resize(t);

    // r0 = b - A * x0
    A->residual(x, b, r);

    // Append norm of initial residual to res
if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
    rr_inner = r.inner_product(r);
if (comm_t) *comm_t += RAPtor_MPI_Wtime();
    norm_r = sqrt(rr_inner);
    res.emplace_back(norm_r);

    // Adjust tolerance
    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    // Initial b_norm (preconditioned)
    z.set_const_value(0.0);
if (precond_t) *precond_t -= RAPtor_MPI_Wtime();
    ml_single->cycle(z, r);
if (precond_t) *precond_t += RAPtor_MPI_Wtime();

    // Perform first iteration outside loop
    // to reduce control flow instructions
    iter = 0;

    // W = T(M^-1 * r0)
    z.split_contig(*W, t, A->partition->first_local_row);

    // A-orthonormalize W
    CGS(A, *W);

    // alpha = W^T * r
    W->mult_T(r, alpha);

    // x = x + W * alpha
    W->mult(alpha, Wa);
    x.axpy(Wa, 1.0);

    // r = r - A * W * alpha
    A->mult(Wa, T);
    r.axpy(T, -1.0);

    // Update norm of residual and iteration
if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
    rr_inner = r.inner_product(r);
if (comm_t) *comm_t += RAPtor_MPI_Wtime();
    norm_r = sqrt(rr_inner);
    res.emplace_back(norm_r);
    iter++;

    // Resize z to be used again during preconditioning
    z.local->b_vecs = t;
    z.resize(b.global_n, b.local_n);

    // Main SRECG Loop
    while (norm_r > tol && iter < max_iter)
    {
        if (iter > 1)
        {
            // Update Wk_1 and Wk_2
            Wk_2->copy(*Wk_1);
            Wk_1->copy(*W);

            // W = A * W
            A->mult(*W, *W_temp);

            // W = M^-1 * A * W
            z.set_const_value(0.0);
        if (precond_t) *precond_t -= RAPtor_MPI_Wtime();
            ml->cycle(z, *W_temp);
        if (precond_t) *precond_t += RAPtor_MPI_Wtime();

            // A-orthonormalize W against previous vectors
            BCGS(A, *Wk_1, *Wk_2, *W_temp);
        }
        else
        {
            // Update Wk_1
            Wk_1->copy(*W);

            // W = A * W
            A->mult(*W, *W_temp);

            // W = M^-1 * A * W
            z.set_const_value(0.0);
        if (precond_t) *precond_t -= RAPtor_MPI_Wtime();
            ml->cycle(z, *W_temp);
        if (precond_t) *precond_t += RAPtor_MPI_Wtime();

            // A-orthonormalize W against previous vectors
            BCGS(A, *Wk_1, *W_temp);
        }

        W->copy(*W_temp);

        // A-orthonormalize W
        CGS(A, *W);

        // alpha = W^T * r
        W->mult_T(r, alpha);

        // x = x + W * alpha
        W->mult(alpha, Wa);
        x.axpy(Wa, 1.0);

        // r = r - A * W * alpha
        A->mult(Wa, T);
        r.axpy(T, -1.0);

        // Update norm of residual and iteration
    if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
        rr_inner = r.inner_product(r);
    if (comm_t) *comm_t += RAPtor_MPI_Wtime();
        norm_r = sqrt(rr_inner);
        res.emplace_back(norm_r);
        iter++;
    }

    if (rank == 0)
    {
        if (iter == max_iter)
        {
            printf("Max Iterations Reached.\n");
        }
        else
        {
            printf("%d Iteration required to converge\n", iter);
        }
        printf("Relative Residual: %lg\n\n", res[iter-1]);
    }

    delete W;
    delete Wk_1;
    delete Wk_2;
    delete W_temp;

    return;
}

