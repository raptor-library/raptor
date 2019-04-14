// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "krylov/par_cg_openmp.hpp"
#include "multilevel/par_multilevel.hpp"
#include "aorthonormalization/par_cgs_timed.hpp"

using namespace raptor;

/* Storing values in times vector: times[allred times, pt2pt times, comp times]*/
void CG_omp(ParCSRMatrix* A, ParVector& x, ParVector& b, aligned_vector<double>& times, aligned_vector<double>& res, double tol, int max_iter)
{
    double start, stop;

    start = MPI_Wtime();        

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
    
    stop = MPI_Wtime();

    times[2] += (stop - start);

    // r0 = b - A * x0
    A->residual_timed(x, b, r, times); 
    
    // p0 = r0
    start = MPI_Wtime();
    p.copy(r);
    stop = MPI_Wtime();
    times[2] += (stop - start);
    
    rr_inner = r.inner_product_timed(r, times);
    
    start = MPI_Wtime();

    norm_r = sqrt(rr_inner);
    res.emplace_back(norm_r / b_norm);

    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    // How often should r be recomputed
    recompute_r = 8;
    iter = 0;
    
    stop = MPI_Wtime();
    times[2] += (stop - start);

    // Main CG Loop
    while (norm_r > tol && iter < max_iter)
    {
        // alpha_i = (r_i, r_i) / (A*p_i, p_i)
        A->mult_timed(p, Ap, times);
        App_inner = Ap.inner_product_timed(p, times);

        start = MPI_Wtime();
        times[2] += (stop - start);
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
        stop = MPI_Wtime();
        times[2] += (stop - start);

        // x_{i+1} = x_i + alpha_i * p_i
        if ((iter % recompute_r) && iter > 0)
        {
            start = MPI_Wtime();
            r.axpy(Ap, -1.0*alpha);
            stop = MPI_Wtime();
            times[2] += (stop - start);
        }
        else
        {
            A->residual_timed(x, b, r, times);
        }

        // beta_i = (r_{i+1}, r_{i+1}) / (r_i, r_i)
        next_inner = r.inner_product_timed(r, times);

        start = MPI_Wtime();
        beta = next_inner / rr_inner;

        // p_{i+1} = r_{i+1} + beta_i * p_i
        p.scale(beta);
        p.axpy(r, 1.0);

        // Update next inner product
        rr_inner = next_inner;
        norm_r = sqrt(rr_inner);
        res.emplace_back(norm_r / b_norm);

        iter++;
            
        stop = MPI_Wtime();
        times[2] += (stop - start);
    }

    return;
}

/* Storing values in times vector: times[allred times, pt2pt times, comp times, aortho times]*/
void SRECG_omp(ParCSRMatrix* A, ParVector& x, ParVector& b, int t, aligned_vector<double>& times, aligned_vector<double>& res, double tol, int max_iter)
{
    double start, stop;

    start = MPI_Wtime();       
 
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
    
    stop = MPI_Wtime();
    times[2] += (stop - start);

    // r0 = b - A * x0
    A->residual_timed(x, b, r, times);

    // Append norm of initial residual to res
    rr_inner = r.inner_product_timed(r, times);

    start = MPI_Wtime();        
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
    
    stop = MPI_Wtime();
    times[2] += (stop - start);

    // A-orthonormalize W
    //start = MPI_Wtime();        
    CGS(A, *W, times);
    //stop = MPI_Wtime();
    //times[3] += (stop - start);
    
    // alpha = W^T * r
    W->mult_T_timed(r, alpha, times);     /* ------------ Profile comm and comp in method ----------- */

    start = MPI_Wtime();        
    // x = x + W * alpha
    W->mult(alpha, Wa);
    x.axpy(Wa, 1.0);
    stop = MPI_Wtime();
    times[2] += (stop - start);

    // r = r - A * W * alpha
    A->mult_timed(Wa, T, times);
    
    start = MPI_Wtime();        
    r.axpy(T, -1.0);
    stop = MPI_Wtime();
    times[2] += (stop - start);

    // Update norm of residual and iteration
    rr_inner = r.inner_product_timed(r, times);

    start = MPI_Wtime();        

    norm_r = sqrt(rr_inner);
    res.emplace_back(norm_r);
    iter++;

    stop = MPI_Wtime();
    times[2] += (stop - start);
    
    // Main SRECG Loop
    while (norm_r > tol && iter < max_iter)
    {
        // A-orthonormalize W against previous vectors
        if (iter > 1)
        {
            // Update Wk_1 and Wk_2
            Wk_2->copy(*Wk_1);
            //Wk_2 = Wk_1;
            Wk_1->copy(*W);
            //Wk_1 = W;

            // W = A * W
            A->mult_timed(*W, *W_temp, times);

            //start = MPI_Wtime();        
            BCGS(A, *Wk_1, *Wk_2, *W_temp, times);
            //stop = MPI_Wtime();
            //times[3] += (stop - start);
        }
        else
        {
            // Update Wk_1
            Wk_1->copy(*W);
            //Wk_1 = W;

            // W = A * W
            A->mult_timed(*W, *W_temp, times);

            //start = MPI_Wtime();        
            BCGS(A, *Wk_1, *W_temp, times);
            //stop = MPI_Wtime();
            //times[3] += (stop - start);
        }

        W->copy(*W_temp);
        //W = W_temp;

        // A-orthonormalize W
        //start = MPI_Wtime();        
        CGS(A, *W, times);
        //stop = MPI_Wtime();
        //times[3] += (stop - start);

        // alpha = W^T * r
        W->mult_T_timed(r, alpha, times); /* ------------ Profile comm and comp in method ----------- */

        start = MPI_Wtime();        
        // x = x + W * alpha
        W->mult(alpha, Wa);
        x.axpy(Wa, 1.0);
        stop = MPI_Wtime();
        times[2] += (stop - start);

        // r = r - A * W * alpha
        A->mult_timed(Wa, T, times);

        start = MPI_Wtime();        
        r.axpy(T, -1.0);
        stop = MPI_Wtime();
        times[2] += (stop - start);

        // Update norm of residual and iteration
        rr_inner = r.inner_product_timed(r, times);
    
        start = MPI_Wtime();        
        norm_r = sqrt(rr_inner);
        res.emplace_back(norm_r);
        iter++;
        stop = MPI_Wtime();
        times[2] += (stop - start);
    }

    delete W;
    delete Wk_1;
    delete Wk_2;
    delete W_temp;
    return;
}
