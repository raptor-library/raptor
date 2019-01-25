// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "aorthonormalization/par_cgs_timed.hpp"

using namespace raptor;

/*******************************************************
*** Block Classical Gram-Schmidt for A-Orthonormaliztion
*** of P against vectors in Q
***
*** ALG 17 IN APPENDIX
********************************************************/
void BCGS(ParCSRMatrix* A, ParBVector& Q1, ParBVector& Q2, ParBVector& P, aligned_vector<double>& times)
{
    double start, stop;

    start = MPI_Wtime();

    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int t = P.local->b_vecs;
    double *inner_prods = new double[t];
    double temp;
    
    ParBVector W(P.global_n, P.local_n, P.first_local, t);
    ParBVector Q(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, t);
    BVector B(Q.local->b_vecs, t);
   
    // Form Q
    Q.copy(Q1);
    Q.append(Q2);

    stop = MPI_Wtime();
    times[2] += (stop - start);
    
    // W = A * P    
    A->mult_timed(P, W, times);
    
    // P = P - Q * (Q^T * W)
    Q.mult_T_timed(W, B, times);

    start = MPI_Wtime();
    Q.mult(B, W);

    P.axpy(W, -1.0);
    stop = MPI_Wtime();
    times[2] += (stop - start);

    // W = A * P
    A->mult_timed(P, W, times);
    
    // P[:,i]^T W[:,i]
    temp = P.inner_product_timed(W, times, inner_prods);

    start = MPI_Wtime();

    // sqrt(inner_prods[i])
    for (int i = 0; i < t; i++)
    {
        temp = pow(inner_prods[i], 1.0/2.0);
        inner_prods[i] = 1.0/temp;
    } 

    // P[:,i] = P[:,i] / ||P[:,i]||_A
    P.scale(1, inner_prods);    
    
    stop = MPI_Wtime();
    times[2] += (stop - start);

    delete inner_prods;

    return;
}

/*******************************************************
*** Block Classical Gram-Schmidt for A-Orthonormaliztion
*** of P against vectors in Q
***
*** ALG 17 IN APPENDIX
********************************************************/
void BCGS(ParCSRMatrix* A, ParBVector& Q1, ParBVector& P, aligned_vector<double>& times)
{
    double start, stop;

    start = MPI_Wtime();

    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int t = P.local->b_vecs;
    double *inner_prods = new double[t];
    double temp;
    
    ParBVector W(P.global_n, P.local_n, P.first_local, t);
    BVector B(Q1.local->b_vecs, t);

    stop = MPI_Wtime();
    times[2] += (stop - start);    
 
    // W = A * P
    A->mult_timed(P, W, times);
    
    // P = P - Q * (Q^T * W)
    Q1.mult_T_timed(W, B, times);

    start = MPI_Wtime();
    Q1.mult(B, W);

    P.axpy(W, -1.0);
    stop = MPI_Wtime();
    times[2] += (stop - start);

    // W = A * P
    A->mult_timed(P, W, times);
    
    // P[:,i]^T W[:,i]
    temp = P.inner_product_timed(W, times, inner_prods);

    start = MPI_Wtime();
    // sqrt(inner_prods[i])
    for (int i = 0; i < t; i++)
    {
        temp = pow(inner_prods[i], 1.0/2.0);
        inner_prods[i] = 1.0/temp;
    } 

    // P[:,i] = P[:,i] / ||P[:,i]||_A
    P.scale(1, inner_prods);    

    delete inner_prods;
    stop = MPI_Wtime();
    times[2] += (stop - start);

    return;
}

/********************************************************
*** Classical Gram-Schmidt for 
*** A-Orthonormalization of vectors in P against one
*** another
***
*** ALG 20 IN APPENDIX
********************************************************/
void CGS(ParCSRMatrix* A, ParBVector& P, aligned_vector<double>& times)
{
    double start, stop;

    start = MPI_Wtime();    

    int t = P.local->b_vecs;
    double inner_prod;
    ParBVector W(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, t);
    ParBVector T1(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, t);
    ParBVector T2(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, t);

    stop = MPI_Wtime();
    times[2] += (stop - start);

    // W = A * P
    A->mult_timed(P, W, times);

    for (int i = 0; i < t; i++)
    {
        for (int j = 0; j < i; j++)
        {
            // P[:,i] = P[:,i] - (P[:,j]^T * A * P[:,i]) * P[:,j]
            inner_prod = P.inner_product_timed(W, j, i, times); // NEED TO CREATE A TIMED VERSION OF THIS
            /*if (inner_prod < zero_tol)
            {
                printf("Linearly dependent vectors.\n");
                exit(-1);
                //inner_prod = zero_tol;
            }*/
            //printf("P_i W_j innerprod %lg\n", inner_prod);
            start = MPI_Wtime();
            P.axpy_ij(P, i, j, -1.0*inner_prod); 
            stop = MPI_Wtime();
            times[2] += (stop - start);
            //printf("---------P\n");
            //P.local->print();
        }

        start = MPI_Wtime();
        // Just multiply ith column of P by A
        T1.set_const_value(0.0);
        for (int k = 0; k < T1.local_n; k++)
        {
            T1.local->values[i*T1.local_n + k] = P.local->values[i*P.local_n + k];
        }
        stop = MPI_Wtime();
        times[2] += (stop - start);

        A->mult_timed(T1, T2, times);

        // P[:,i]^T A P[:,i]
        //inner_prod = P.inner_product(T2, inner_prods);
        inner_prod = P.inner_product_timed(T2, i, i, times); // NEED TO CREATE A TIMED VERSION OF THIS
        //printf("P^T A P inner_prod %lg\n", inner_prod);

        start = MPI_Wtime();
        // sqrt(inner_prod)
        inner_prod = pow(inner_prod, 1.0/2.0);

        // P[:,i] = P[:,i] / ||P[:,i]||_A
        //P.scale(1, inner_prods);
        for (int k = 0; k < P.local_n; k++) P.local->values[i*P.local_n + k] *= 1.0/inner_prod;
        stop = MPI_Wtime();
        times[2] += (stop - start);
    }
    
    /*double *inner_prods = new double[t];
    inner_prod = P.inner_product(P, inner_prods);
    for (int i = 0; i < t; i++) inner_prods[i] = 1.0/inner_prods[i];
    P.scale(1, inner_prods);

    delete inner_prods;*/

    return;
}
