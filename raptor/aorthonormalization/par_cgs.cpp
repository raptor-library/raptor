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
    int t = P.local->b_vecs;
    double *inner_prods = new double[t];
    double temp;
    
    ParBVector W(P.global_n, P.local_n, P.first_local, t);
    ParBVector Q(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, t);
    BVector B(Q.local->b_vecs, t);

    // Form Q
    Q.append(Q1);
    Q.append(Q2);

    // Performed in SRE-CG
    // W = A * P    
    //A->mult(P, W);

    W.copy(P);
    
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
        inner_prods[i] = 1/temp;
    } 

    // P[:,i] = P[:,i] / ||P[:,i]||_A
    P.scale(1, inner_prods);    

    delete inner_prods;

    return;
}

/********************************************************
*** QR with Classical Gram-Schmidt for 
*** A-Orthonormalization of vectors in P against one
*** another
***
*** ALG 20 IN APPENDIX
********************************************************/
void QR(ParCSRMatrix* A, ParBVector& P)
{
    int t = P.local->b_vecs;
    ParBVector W(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, t);

    // W = A * P
    A->mult(P, W);

    return;
}
