#include "aorthonormalization/par_cgs.hpp"

using namespace raptor;

/********************************************************
*** Block Classical Gram-Schmidt for A-Orthonormalization
*** of P against vectors in Q
***
*** ALG 17 IN APPENDIX
*********************************************************/
void BCGS(ParCSRMatrix* A, ParBVector& Q1, ParBVector& Q2, ParBVector& P, double* comp_t, bool tap)
{
    int t = P.local->b_vecs;
    double *inner_prods = new double[t];
    double temp;

    ParBVector W(P.global_n, P.local_n, t);
    ParBVector Q(A->global_num_rows, A->local_num_rows, t);
    BVector B(Q.local->b_vecs, t);

    // Form Q
    Q.copy(Q1);
    Q.append(Q2);

    // W = A * P
    A->mult(P, W, tap);

    // P = P - Q * (Q^T * W)
    Q.mult_T(W, B);
if (comp_t) *comp_t -= RAPtor_MPI_Wtime();
    Q.mult(B, W);
    P.axpy(W, -1.0);
if (comp_t) *comp_t += RAPtor_MPI_Wtime();
    
    // W = A * P
    A->mult(P, W, tap);
    
    // P[:,i]^T W[:,i]
    temp = P.inner_product(W, inner_prods);
    
    // sqrt(inner_prods[i])
if (comp_t) *comp_t -= RAPtor_MPI_Wtime();
    for (int i = 0; i < t; i++)
    {
        temp = pow(inner_prods[i], 1.0/2.0);
        inner_prods[i] = 1.0/temp;
    }
if (comp_t) *comp_t += RAPtor_MPI_Wtime();

    // P[:,i] = P[:,i] / ||P[:,i]||_A
if (comp_t) *comp_t -= RAPtor_MPI_Wtime();
    P.scale(1, inner_prods);
if (comp_t) *comp_t += RAPtor_MPI_Wtime();

    delete inner_prods;

    return;
}

/********************************************************
*** Block Classical Gram-Schmidt for A-Orthonormalization
*** of P against vectors in Q
***
*** ALG 17 IN APPENDIX
*********************************************************/
void BCGS(ParCSRMatrix* A, ParBVector& Q1, ParBVector& P, double* comp_t, bool tap)
{
    int t = P.local->b_vecs;
    double *inner_prods = new double[t];
    double temp;

    ParBVector W(P.global_n, P.local_n, t);
    BVector B(Q1.local->b_vecs, t);

    // W = A * P
    A->mult(P, W, tap);

    // P = P - Q * (Q^T * W)
    Q1.mult_T(W, B);
if (comp_t) *comp_t -= RAPtor_MPI_Wtime();
    Q1.mult(B, W);
    P.axpy(W, -1.0);
if (comp_t) *comp_t += RAPtor_MPI_Wtime();

    // W = A * P
    A->mult(P, W, tap);

    // P[:,i]^T w[:,i]
    temp = P.inner_product(W, inner_prods);

    // sqrt(inner_prods[i])
if (comp_t) *comp_t -= RAPtor_MPI_Wtime();
    for (int i = 0; i < t; i++)
    {
        temp = pow(inner_prods[i], 1.0/2.0);
        inner_prods[i] = 1.0/temp;
    }
if (comp_t) *comp_t += RAPtor_MPI_Wtime();

    // P[:,i] = P[:,i] / ||P[:,i]||_A
if (comp_t) *comp_t -= RAPtor_MPI_Wtime();
    P.scale(1, inner_prods);
if (comp_t) *comp_t += RAPtor_MPI_Wtime();

    delete inner_prods;

    return;
}

/*************************************************
*** Classical Gram-Schmidt for A-Orthnormalization 
*** of vectors in P against one another
***
*** ALG 20 IN APPENDIX
**************************************************/
void CGS(ParCSRMatrix* A, ParBVector& P, double* comp_t, bool tap)
{
    int t = P.local->b_vecs;
    double inner_prod;
    ParBVector W(A->global_num_rows, A->local_num_rows, t);
    ParBVector T1(A->global_num_rows, A->local_num_rows, t);
    ParBVector T2(A->global_num_rows, A->local_num_rows, t);

    // W = A * P
    A->mult(P, W, tap);

    for (int i = 0; i < t; i++)
    {
        for (int j = 0; j < i; j++)
        {
            // P[:,i] = P[:,i] - (P[:,j]^T * A * P[:,i]) * P[:,j]
            inner_prod = P.inner_product(W, j, i);
        if (comp_t) *comp_t -= RAPtor_MPI_Wtime();
            P.axpy_ij(P, i, j, -1.0*inner_prod);
        if (comp_t) *comp_t += RAPtor_MPI_Wtime();
        }

        // Just multiply ith column of P by A
        T1.set_const_value(0.0);
        for (int k = 0; k < T1.local_n; k++)
        {
            T1.local->values[i*T1.local_n + k] = P.local->values[i*P.local_n + k];
        }

        A->mult(T1, T2, tap);

        // P[:,i]^T A P[:,i]
        inner_prod = P.inner_product(T2, i, i);
        inner_prod = pow(inner_prod, 1.0/2.0);

        // P[:,i] = P[:,i] / ||P{:,i}||_A
    if (comp_t) *comp_t -= RAPtor_MPI_Wtime();
        for (int k = 0; k < P.local_n; k++) P.local->values[i*P.local_n + k] *= 1.0/inner_prod;
    if (comp_t) *comp_t += RAPtor_MPI_Wtime();
    }
    return;
}

/*************************************************
*** Modified Gram-Schmidt for A-Orthnormalization 
*** of vectors in P against one another
**************************************************/
void MGS(ParCSRMatrix* A, ParBVector& P)
{
    int t = P.local->b_vecs;
    double *inner_prods = new double[t];
    double inner_prod;
    double temp;

    ParBVector W(A->global_num_rows, A->local_num_rows, t);
    ParBVector T1(A->global_num_rows, A->local_num_rows, t);
    ParBVector T2(A->global_num_rows, A->local_num_rows, t);
    A->mult(P, W);

    for (int i = 0; i < t; i++)
    {
        inner_prod = P.inner_product(W, i, i);
        inner_prod = pow(inner_prod, 1.0/2.0);
        for (int k = 0; k < P.local_n; k++) P.local->values[i*P.local_n + k] *= 1.0/inner_prod;
        for (int j = i+1; j < t; j++)
        {
            // Just multiply ith column of P by A
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
