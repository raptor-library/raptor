// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "aorthonormalization/par_cgs.hpp"

using namespace raptor;

/*******************************************************
*** Block Classical Gram-Schmidt for A-Orthonormaliztion
*** of P against vectors in Q
***
*** ALG 17 IN APPENDIX
********************************************************/
void BCGS(ParCSRMatrix* A, ParBVector& Q1, ParBVector& Q2, ParBVector& P)
{
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
    
    // W = A * P    
    A->mult(P, W);
    
    // P = P - Q * (Q^T * W)
    Q.mult_T(W, B);
    Q.mult(B, W);
    P.axpy(W, -1.0);

    // W = A * P
    A->mult(P, W);
    
    // P[:,i]^T W[:,i]
    temp = P.inner_product(W, inner_prods);

    // sqrt(inner_prods[i])
    for (int i = 0; i < t; i++)
    {
        temp = pow(inner_prods[i], 1.0/2.0);
        inner_prods[i] = 1.0/temp;
    } 

    // P[:,i] = P[:,i] / ||P[:,i]||_A
    P.scale(1, inner_prods);    

    delete inner_prods;

    return;
}

/*******************************************************
*** Block Classical Gram-Schmidt for A-Orthonormaliztion
*** of P against vectors in Q
***
*** ALG 17 IN APPENDIX
********************************************************/
void BCGS(ParCSRMatrix* A, ParBVector& Q1, ParBVector& P)
{
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int t = P.local->b_vecs;
    double *inner_prods = new double[t];
    double temp;
    
    ParBVector W(P.global_n, P.local_n, P.first_local, t);
    BVector B(Q1.local->b_vecs, t);
 
    // W = A * P
    A->mult(P, W);
    
    // P = P - Q * (Q^T * W)
    Q1.mult_T(W, B);
    Q1.mult(B, W);
    P.axpy(W, -1.0);

    // W = A * P
    A->mult(P, W);
    
    // P[:,i]^T W[:,i]
    temp = P.inner_product(W, inner_prods);

    // sqrt(inner_prods[i])
    for (int i = 0; i < t; i++)
    {
        temp = pow(inner_prods[i], 1.0/2.0);
        inner_prods[i] = 1.0/temp;
    } 

    // P[:,i] = P[:,i] / ||P[:,i]||_A
    P.scale(1, inner_prods);    

    delete inner_prods;

    return;
}

/********************************************************
*** Classical Gram-Schmidt for 
*** A-Orthonormalization of vectors in P against one
*** another
***
*** ALG 20 IN APPENDIX
********************************************************/
void CGS(ParCSRMatrix* A, ParBVector& P)
{
    int t = P.local->b_vecs;
    double inner_prod;
    ParBVector W(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, t);
    ParBVector T1(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, t);
    ParBVector T2(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, t);

    // W = A * P
    A->mult(P, W);

    for (int i = 0; i < t; i++)
    {
        for (int j = 0; j < i; j++)
        {
            // P[:,i] = P[:,i] - (P[:,j]^T * A * P[:,i]) * P[:,j]
            inner_prod = P.inner_product(W, j, i);
            /*if (inner_prod < zero_tol)
            {
                printf("Linearly dependent vectors.\n");
                exit(-1);
                //inner_prod = zero_tol;
            }*/
            //printf("P_i W_j innerprod %lg\n", inner_prod);
            P.axpy_ij(P, i, j, -1.0*inner_prod);
            //printf("---------P\n");
            //P.local->print();
        }

        // Just multiply ith column of P by A
        T1.set_const_value(0.0);
        for (int k = 0; k < T1.local_n; k++)
        {
            T1.local->values[i*T1.local_n + k] = P.local->values[i*P.local_n + k];
        }

        A->mult(T1, T2);

        // P[:,i]^T A P[:,i]
        //inner_prod = P.inner_product(T2, inner_prods);
        inner_prod = P.inner_product(T2, i, i);
        //printf("P^T A P inner_prod %lg\n", inner_prod);

        // sqrt(inner_prod)
        inner_prod = pow(inner_prod, 1.0/2.0);

        // P[:,i] = P[:,i] / ||P[:,i]||_A
        //P.scale(1, inner_prods);
        for (int k = 0; k < P.local_n; k++) P.local->values[i*P.local_n + k] *= 1.0/inner_prod;
    }
    
    /*double *inner_prods = new double[t];
    inner_prod = P.inner_product(P, inner_prods);
    for (int i = 0; i < t; i++) inner_prods[i] = 1.0/inner_prods[i];
    P.scale(1, inner_prods);

    delete inner_prods;*/

    return;
}

/*******************************************************
*** Modified Gram-Schmidt for A-Orthonormaliztion
*** of P's vectors 
***
********************************************************/
void MGS(ParCSRMatrix* A, ParBVector& P)
{
    int t = P.local->b_vecs;
    double *inner_prods = new double[t];
    double inner_prod;
    double temp;
    
    ParBVector W(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, t);
    ParBVector T1(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, t);
    ParBVector T2(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, t);
    A->mult(P, W);

    for (int i = 0; i < t; i++)
    {
        inner_prod = P.inner_product(W, i, i);
        inner_prod = pow(inner_prod, 1.0/2.0);
        for (int k = 0; k < P.local_n; k++) P.local->values[i*P.local_n + k] *= 1/inner_prod;
        for (int j = i+1; j < t; j++)
        {
            // Just multiply jth column of P by A
            T1.set_const_value(0.0);
            for (int k = 0; k < T1.local_n; k++)
            {
                T1.local->values[j*T1.local_n + k] = P.local->values[j*P.local_n + k];
            }

            A->mult(T1, T2);
            inner_prod = P.inner_product(T2, i, j);
            P.axpy_ij(P, j, i, -1.0*inner_prod);
        }
    }    

    delete inner_prods;

    return;
}
