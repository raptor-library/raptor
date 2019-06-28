// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "krylov/par_cg.hpp"
#include "multilevel/par_multilevel.hpp"
#include "aorthonormalization/par_cgs.hpp"

using namespace raptor;

void CG(ParCSRMatrix* A, ParVector& x, ParVector& b, aligned_vector<double>& res, double tol, int max_iter)
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
    double b_norm = b.norm(2);
    if (b_norm < zero_tol) b_norm = 1.0;

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
        res.emplace_back(norm_r / b_norm);

        iter++;
    }

    /*if (rank == 0)
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
    }*/

    return;
}


void PCG(ParCSRMatrix* A, ParMultilevel* ml, ParVector& x, ParVector& b, aligned_vector<double>& res, double tol, int max_iter)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ParVector r;
    ParVector z;
    ParVector p;
    ParVector Ap;

    int iter;
    int recompute_r = 4;
    bool full_r;
    data_t alpha, beta;
    data_t b_inner, rz_inner, next_inner, App_inner;
    double norm_b, norm_rz;

    if (max_iter <= 0)
    {
        max_iter = ((int)(1.3*b.global_n)) + 2;
    }

    // Fixed Constructors
    r.resize(b.global_n, b.local_n, b.first_local);
    z.resize(b.global_n, b.local_n, b.first_local);
    p.resize(b.global_n, b.local_n, b.first_local);
    Ap.resize(b.global_n, b.local_n, b.first_local);

    // Initial b_norm (preconditioned)
    z.set_const_value(0.0);
    ml->cycle(z, b);
    b_inner = b.inner_product(z);
    norm_b = sqrt(b_inner);
    if (norm_b > zero_tol)
    {
        tol = tol * norm_b;
    }

    // r0 = b - A * x0
    A->residual(x, b, r);

    // z0 = M^{-1}r0
    z.set_const_value(0.0);
    ml->cycle(z, r);

    // p0 = z0
    p.copy(z);

    // <r, z>
    rz_inner = r.inner_product(z);
    norm_rz = sqrt(rz_inner);
    res.emplace_back(norm_rz);

    recompute_r = 8;
    iter = 0;

    // Main CG Loop
    while (iter < max_iter)
    {
        iter++;

        // alpha_i = (r_i, z_i) / (A*p_i, p_i)
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

        full_r = recompute_r && iter % recompute_r == 0;

        if (full_r)
        {
            A->residual(x, b, r);
        }
        else
        {
            r.axpy(Ap, -1.0*alpha);
        }

        // z_{j+1} = M^{-1}r_{j+1}
        z.set_const_value(0.0);
        ml->cycle(z, r);

        // beta_i = (r_{i+1}, z_{i+1}) / (r_i, z_i)
        next_inner = r.inner_product(z);
        beta = next_inner / rz_inner;

        res.emplace_back(next_inner/b_inner);
        if (next_inner < tol) break;

        // p_{i+1} = z_{i+1} + beta_i * p_i
        if (full_r)
        {
            p.copy(z);
        }
        else
        {
            p.scale(beta);
            p.axpy(z, 1.0);
        }

        // Update next inner product
        rz_inner = next_inner;
    }

    /*if (rank == 0)
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
    }*/

    return;
}

void SRECG(ParCSRMatrix* A, ParVector& x, ParVector& b, int t, aligned_vector<double>& res, double tol, int max_iter)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    ParVector r;
    ParBVector *W = new ParBVector(A->global_num_cols, A->local_num_rows, A->partition->first_local_row, t);
    ParBVector *W_temp = new ParBVector(A->global_num_cols, A->local_num_rows, A->partition->first_local_row, t);
    ParBVector *Wk_1 = new ParBVector(A->global_num_cols, A->local_num_rows, A->partition->first_local_row, t);
    ParBVector *Wk_2 = new ParBVector(A->global_num_cols, A->local_num_rows, A->partition->first_local_row, t);
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
    r.resize(b.global_n, b.local_n, b.first_local);
    Wa.resize(x.global_n, x.local_n, x.first_local);
    T.resize(x.global_n, x.local_n, x.first_local);
    alpha.resize(t);

    // r0 = b - A * x0
    A->residual(x, b, r);

    // Append norm of initial residual to res
    rr_inner = r.inner_product(r);
    norm_r = sqrt(rr_inner);
    res.emplace_back(norm_r);

    // Adjust tolerance
    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    // Perform first iteration outside loop
    // to reduce control flow instructions
    iter = 0;

    // W = T(r0)
    r.split_contig(*W, t);

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
    rr_inner = r.inner_product(r);
    norm_r = sqrt(rr_inner);
    res.emplace_back(norm_r);
    iter++;
    
    // Main SRECG Loop
    while (norm_r > tol && iter < max_iter)
    {
        /*if (iter == 0)
        {
            // W = T(r0)
            r.split_contig(W, t);

            // A-orthonormalize W
            CGS(A, W);
        }
        else
        {*/
            // Update Wk_1 and Wk_2
            /*Wk_2.copy(Wk_1);
            Wk_1.copy(W);

            // W = A * W
            A->mult(W, W_temp);*/

            // A-orthonormalize W against previous vectors
            if (iter > 1)
            {
                // Update Wk_1 and Wk_2
                Wk_2->copy(*Wk_1);
                //Wk_2 = Wk_1;
                Wk_1->copy(*W);
                //Wk_1 = W;

                // W = A * W
                A->mult(*W, *W_temp);

                BCGS(A, *Wk_1, *Wk_2, *W_temp);
            }
            else
            {
                // Update Wk_1
                Wk_1->copy(*W);
                //Wk_1 = W;

                // W = A * W
                A->mult(*W, *W_temp);

                BCGS(A, *Wk_1, *W_temp);
            }

            W->copy(*W_temp);
            //W = W_temp;

            // A-orthonormalize W
            CGS(A, *W);
        //}

        // alpha = W^T * r
        W->mult_T(r, alpha);

        // x = x + W * alpha
        W->mult(alpha, Wa);
        x.axpy(Wa, 1.0);

        // r = r - A * W * alpha
        A->mult(Wa, T);
        r.axpy(T, -1.0);

        // Update norm of residual and iteration
        rr_inner = r.inner_product(r);
        norm_r = sqrt(rr_inner);
        res.emplace_back(norm_r);
        iter++;
    }

    /*if (rank == 0)
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
    }*/

    delete W;
    delete Wk_1;
    delete Wk_2;
    delete W_temp;
    return;
}

void PSRECG(ParCSRMatrix* A, ParMultilevel *ml, ParVector& x, ParVector& b, int t, aligned_vector<double>& res, double tol, int max_iter)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    ParVector r;
    ParVector z;
    ParBVector *W = new ParBVector(A->global_num_cols, A->local_num_rows, A->partition->first_local_row, t);
    ParBVector *W_temp = new ParBVector(A->global_num_cols, A->local_num_rows, A->partition->first_local_row, t);
    ParBVector *Wk_1 = new ParBVector(A->global_num_cols, A->local_num_rows, A->partition->first_local_row, t);
    ParBVector *Wk_2 = new ParBVector(A->global_num_cols, A->local_num_rows, A->partition->first_local_row, t);
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
    r.resize(b.global_n, b.local_n, b.first_local);
    z.resize(b.global_n, b.local_n, b.first_local);
    Wa.resize(x.global_n, x.local_n, x.first_local);
    T.resize(x.global_n, x.local_n, x.first_local);
    alpha.resize(t);

    // r0 = b - A * x0
    A->residual(x, b, r);

    // Append norm of initial residual to res
    rr_inner = r.inner_product(r);
    norm_r = sqrt(rr_inner);
    res.emplace_back(norm_r);

    // Adjust tolerance
    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    // Initial residual preconditioned before splitting
    z.set_const_value(0.0);
    ml->cycle(z, r);

    // Perform first iteration outside loop
    // to reduce control flow instructions
    iter = 0;

    // W = T(M^-1 * r0)
    z.split_contig(*W, t);

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
    rr_inner = r.inner_product(r);
    norm_r = sqrt(rr_inner);
    res.emplace_back(norm_r);
    iter++;
    
    // Resize z to be used again during preconditioning
    z.local->b_vecs = t;
    z.resize(b.global_n, b.local_n, b.first_local);
    
    // Main SRECG Loop
    while (norm_r > tol && iter < max_iter)
    {
        if (iter > 1)
        {
            // Update Wk_1 and Wk_2
            Wk_2->copy(*Wk_1);
            //Wk_2 = Wk_1;
            Wk_1->copy(*W);
            //Wk_1 = W;

            // W = A * W
            A->mult(*W, *W_temp);

            // W = M^-1 * A * W
            z.set_const_value(0.0);
            ml->cycle(z, *W_temp);

            // A-orthonormalize W against previous vectors
            BCGS(A, *Wk_1, *Wk_2, *W_temp);
        }
        else
        {
            // Update Wk_1
            Wk_1->copy(*W);
            //Wk_1 = W;

            // W = A * W
            A->mult(*W, *W_temp);
           
            printf("before cycle\n");
            // W = M^-1 * A * W
            z.set_const_value(0.0);
            ml->cycle(z, *W_temp);
            printf("after cycle\n");

            // A-orthonormalize W against previous vectors
            BCGS(A, *Wk_1, *W_temp);
        }

        W->copy(*W_temp);
        //W = W_temp;

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
        rr_inner = r.inner_product(r);
        norm_r = sqrt(rr_inner);
        res.emplace_back(norm_r);
        iter++;
    }

    /*if (rank == 0)
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
    }*/
    delete W;
    delete Wk_1;
    delete Wk_2;
    delete W_temp;
    return;
}
