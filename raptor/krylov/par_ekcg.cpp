// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "krylov/par_ekcg.hpp"
#include "multilevel/par_multilevel.hpp"
#include "aorthonormalization/par_cgs.hpp"

using namespace raptor;

/* comm_t measures all reduce time for algorithm */
void EKCG(ParCSRMatrix* A, ParVector& x, ParVector& b, int t, aligned_vector<double>& res, 
        double tol, int max_iter, double* comp_t, double* bv_t, bool tap)
{
    int rank;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);

    ParVector r;
    ParBVector *T = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *R = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *Z = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *P = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *P_1 = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *AP = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *AP_1 = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *X = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    Vector alpha;
    Vector gamma;
    Vector rho;

    // Used in lapack routines
    char u = 'U';
    int info;
    char side = 'R';
    char trans = 'N';
    char unit = 'N';
    double alph = 1e0;

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
    gamma.b_vecs = t;
    gamma.resize(t);
    rho.b_vecs = t;
    rho.resize(t);
    alpha.b_vecs = t;
    alpha.resize(t);

    // Initialize columns of x
    X->set_const_value(0.0);

    // r0 = b - A * x0
    if (bv_t) *bv_t -= RAPtor_MPI_Wtime();
    A->residual(x, b, r, tap, comp_t);
    if (bv_t) *bv_t += RAPtor_MPI_Wtime();

    // Append norm of initial residual to res
    rr_inner = r.inner_product(r, NULL, comp_t);
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

    // R = T(r0)
    r.split_contig(*R, t, A->partition->first_local_row);

    // Z = 0 
    Z->copy(*R);

    // P = 0, P_1 = 0
    P->set_const_value(0.0);
    P_1->set_const_value(0.0);

    // Store AP, AP1
    AP->set_const_value(0.0);
    AP_1->set_const_value(0.0);

    iter = 1;
    // Main ECG Loop
    while (true)
    {
        // P = Z * (Z^T * A * Z)^{-1/2}
        // Cholesky of (Z^T * A * Z)
    if (bv_t) *bv_t -= RAPtor_MPI_Wtime();
        A->mult(*Z, *AP, comp_t);
    if (bv_t) *bv_t += RAPtor_MPI_Wtime();
        Z->mult_T(*AP, rho, comp_t);

        // Cholesky of Z^T A Z
    if (comp_t) *comp_t -= RAPtor_MPI_Wtime();
        dpotrf_(&u, &t, rho.values.data(), &t, &info);
    if (comp_t) *comp_t += RAPtor_MPI_Wtime();
        
        // Solve for P
        P->copy(*Z);
    if (comp_t) *comp_t -= RAPtor_MPI_Wtime();
        dtrsm_(&side, &u, &trans, &unit, &(P->local->num_values), &t, &alph, rho.values.data(),
                &t, P->local->values.data(), &(P->local->num_values));
    if (comp_t) *comp_t += RAPtor_MPI_Wtime();
        // Solve for AP
    if (comp_t) *comp_t -= RAPtor_MPI_Wtime();
        dtrsm_(&side, &u, &trans, &unit, &(AP->local->num_values), &t, &alph, rho.values.data(),
                &t, AP->local->values.data(), &(AP->local->num_values));
    if (comp_t) *comp_t += RAPtor_MPI_Wtime();

        // alpha = P^T * R 
        P->mult_T(*R, alpha, comp_t);

        // X = X + P * alpha
        P->mult(alpha, *T, comp_t);
        X->axpy(*T, 1.0, comp_t);

        // R = R - A * P * alpha
        AP->mult(alpha, *T, comp_t);
        R->axpy(*T, -1.0, comp_t);

        // Update norm of residual and iteration
        R->sum_cols(r);
        rr_inner = r.inner_product(r, NULL, comp_t);
        norm_r = sqrt(rr_inner);
        res.emplace_back(norm_r);

        if (norm_r < tol || iter > max_iter)
        {
            // Sum X and return
            X->sum_cols(x);

            delete T;
            delete R;
            delete Z;
            delete P;
            delete P_1;
            delete AP;
            delete AP_1;
            delete X;

            return;
        }

        iter++;

        // gamma = (AP)^T * (AP)
        AP->mult_T(*AP, gamma, comp_t);

        // rho = (AP_1)^T * (AP)
        AP_1->mult_T(*AP, rho, comp_t);

        // Z = AP - P * gamma - P_1 * rho
        AP_1->copy(*AP);
        Z->copy(*AP);

        P->mult(gamma, *T, comp_t);
        P_1->mult(rho, *AP, comp_t);
        Z->axpy(*T, -1.0, comp_t);
        Z->axpy(*AP, -1.0, comp_t);
       
        // Copy P for next iteration
        P_1->copy(*P);
    }
}

/* comm_t measures all reduce time for algorithm */
void EKCG_MinComm(ParCSRMatrix* A, ParVector& x, ParVector& b, int t, aligned_vector<double>& res, 
        double tol, int max_iter, double* comp_t, double* bv_t, bool tap)
{
    int rank;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);

    ParVector r;
    ParBVector *T = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *R = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *Z = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *P = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *P_1 = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *AP = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *AP_1 = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *X = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    Vector alpha;
    Vector gamma;
    Vector rho;
    aligned_vector<double> temp;

    // Used in lapack routines
    char u = 'U';
    int info;
    char side = 'R';
    char trans = 'N';
    char unit = 'N';
    double alph = 1e0;

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
    gamma.b_vecs = t;
    gamma.resize(t);
    rho.b_vecs = t;
    rho.resize(t);
    alpha.b_vecs = t;
    alpha.resize(t);

    // Initialize columns of x
    X->set_const_value(0.0);

    // r0 = b - A * x0
    if (bv_t) *bv_t -= RAPtor_MPI_Wtime();
    A->residual(x, b, r, tap, comp_t);
    if (bv_t) *bv_t += RAPtor_MPI_Wtime();

    // Append norm of initial residual to res
    rr_inner = r.inner_product(r, NULL, comp_t);
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

    // R = T(r0)
    r.split_contig(*R, t, A->partition->first_local_row);

    // Z = 0 
    Z->copy(*R);

    // P = 0, P_1 = 0
    P->set_const_value(0.0);
    P_1->set_const_value(0.0);

    // Store AP, AP1
    AP->set_const_value(0.0);
    AP_1->set_const_value(0.0);

    iter = 1;
    // Main ECG Loop
    while (true)
    {
        // P = Z * (Z^T * A * Z)^{-1/2}
        // Cholesky of (Z^T * A * Z)
    if (bv_t) *bv_t -= RAPtor_MPI_Wtime();
        A->mult(*Z, *AP, comp_t);
    if (bv_t) *bv_t += RAPtor_MPI_Wtime();
        Z->mult_T(*AP, rho, comp_t);

        // Cholesky of Z^T A Z
    if (comp_t) *comp_t -= RAPtor_MPI_Wtime();
        dpotrf_(&u, &t, rho.values.data(), &t, &info);
    if (comp_t) *comp_t += RAPtor_MPI_Wtime();
        
        // Solve for P
        P->copy(*Z);
    if (comp_t) *comp_t -= RAPtor_MPI_Wtime();
        dtrsm_(&side, &u, &trans, &unit, &(P->local->num_values), &t, &alph, rho.values.data(),
                &t, P->local->values.data(), &(P->local->num_values));
    if (comp_t) *comp_t += RAPtor_MPI_Wtime();
        // Solve for AP
    if (comp_t) *comp_t -= RAPtor_MPI_Wtime();
        dtrsm_(&side, &u, &trans, &unit, &(AP->local->num_values), &t, &alph, rho.values.data(),
                &t, AP->local->values.data(), &(AP->local->num_values));
    if (comp_t) *comp_t += RAPtor_MPI_Wtime();

        // alpha = P^T * R 
        P->mult_T(*R, alpha, comp_t);

        // X = X + P * alpha
        P->mult(alpha, *T, comp_t);
        X->axpy(*T, 1.0, comp_t);

        // R = R - AP * alpha
        AP->mult(alpha, *T, comp_t);
        R->axpy(*T, -1.0, comp_t);

        // Update norm of residual and iteration
        R->sum_cols(r);
        rr_inner = r.inner_product(r, NULL, comp_t);
        norm_r = sqrt(rr_inner);
        res.emplace_back(norm_r);

        if (norm_r < tol || iter > max_iter)
        {
            // Sum X and return
            X->sum_cols(x);

            delete T;
            delete R;
            delete Z;
            delete P;
            delete P_1;
            delete AP;
            delete AP_1;
            delete X;

            return;
        }

        iter++;

        // gamma = (AP)^T * (AP)
        // rho = (AP_1)^T * (AP)
        // 1. Compute local portion of gamma and rho
        if (comp_t) *comp_t -= RAPtor_MPI_Wtime();
            AP->local->mult_T(*(AP->local), gamma);
            AP_1->local->mult_T(*(AP->local), rho);
        if (comp_t) *comp_t += RAPtor_MPI_Wtime();
        // 2. Reduce gamma and rho in same array
        std::copy(gamma.values.begin(), gamma.values.end(), back_inserter(temp));
        std::copy(rho.values.begin(), rho.values.end(), back_inserter(temp));
        RAPtor_MPI_Allreduce(RAPtor_MPI_IN_PLACE, &(temp[0]), 2*t*t, RAPtor_MPI_DATA_T, RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);
        // 3. Get gamma from beginning of reduction array
        //    Get rho from end of reduction array
        gamma.values.assign(temp.begin(), temp.begin()+(t*t));
        rho.values.assign(temp.begin()+(t*t), temp.end());
        temp.clear();
        // 4. Compute Z as usual 

        //AP->mult_T(*AP, gamma, comp_t);
        //AP_1->mult_T(*AP, rho, comp_t);

        // Z = AP - P * gamma - P_1 * rho
        AP_1->copy(*AP);
        Z->copy(*AP);

        P->mult(gamma, *T, comp_t);
        P_1->mult(rho, *AP, comp_t);
        Z->axpy(*T, -1.0, comp_t);
        Z->axpy(*AP, -1.0, comp_t);
       
        // Copy P for next iteration
        P_1->copy(*P);
    }
}

/* comm_t measures all reduce time for algorithm */
void PEKCG(ParMultilevel* ml_single, ParMultilevel* ml, ParCSRMatrix* A, ParVector& x, ParVector& b, int t,
        aligned_vector<double>& res, double tol, int max_iter, double* precond_t, double* comp_t, bool tap)
{
    int rank;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);

    ParVector r;
    ParBVector *T = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *R = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *Z = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *P = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *P_1 = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *AP = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *MAP = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *AP_1 = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    ParBVector *X = new ParBVector(A->global_num_cols, A->local_num_rows, t);
    Vector alpha;
    Vector gamma;
    Vector rho;

    // Used in lapack routines
    char u = 'U';
    int info;
    char side = 'R';
    char trans = 'N';
    char unit = 'N';
    double alph = 1e0;

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
    gamma.b_vecs = t;
    gamma.resize(t);
    rho.b_vecs = t;
    rho.resize(t);
    alpha.b_vecs = t;
    alpha.resize(t);

    // Initialize columns of x
    X->set_const_value(0.0);

    // r0 = b - A * x0
    A->residual(x, b, r, tap, comp_t);

    // Append norm of initial residual to res
    rr_inner = r.inner_product(r, NULL, comp_t);
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

    // R = T(r0)
    x.set_const_value(0.0);
    ml_single->cycle(x, r);
    x.split_contig(*R, t, A->partition->first_local_row);
    //r.split_contig(*R, t, A->partition->first_local_row);

    // Z = R 
    Z->copy(*R);
    //Z->set_const_value(0.0);

    // Precondition initial residual
    /*if (precond_t) *precond_t -= RAPtor_MPI_Wtime();
        ml->cycle(*Z, *R);
    if (precond_t) *precond_t += RAPtor_MPI_Wtime();*/

    // P = 0, P_1 = 0
    P->set_const_value(0.0);
    P_1->set_const_value(0.0);

    // Store AP, AP1
    AP->set_const_value(0.0);
    AP_1->set_const_value(0.0);

    iter = 1;
    // Main ECG Loop
    while (true)
    {
        // P = Z * (Z^T * A * Z)^{-1/2}
        // Cholesky of (Z^T * A * Z)
        A->mult(*Z, *AP, comp_t);
        Z->mult_T(*AP, rho, comp_t);

        // Cholesky of Z^T A Z
    if (comp_t) *comp_t -= RAPtor_MPI_Wtime();
        dpotrf_(&u, &t, rho.values.data(), &t, &info);
    if (comp_t) *comp_t += RAPtor_MPI_Wtime();
        
        // Solve for P
        P->copy(*Z);
    if (comp_t) *comp_t -= RAPtor_MPI_Wtime();
        dtrsm_(&side, &u, &trans, &unit, &(P->local->num_values), &t, &alph, rho.values.data(),
                &t, P->local->values.data(), &(P->local->num_values));
    if (comp_t) *comp_t += RAPtor_MPI_Wtime();
        // Solve for AP
    if (comp_t) *comp_t -= RAPtor_MPI_Wtime();
        dtrsm_(&side, &u, &trans, &unit, &(AP->local->num_values), &t, &alph, rho.values.data(),
                &t, AP->local->values.data(), &(AP->local->num_values));
    if (comp_t) *comp_t += RAPtor_MPI_Wtime();

        // alpha = P^T * R 
        P->mult_T(*R, alpha, comp_t);

        // X = X + P * alpha
        P->mult(alpha, *T, comp_t);
        X->axpy(*T, 1.0, comp_t);

        // R = R - A * P * alpha
        AP->mult(alpha, *T, comp_t);
        R->axpy(*T, -1.0, comp_t);

        // Update norm of residual and iteration
        R->sum_cols(r);
        rr_inner = r.inner_product(r, NULL, comp_t);
        norm_r = sqrt(rr_inner);
        res.emplace_back(norm_r);

        if (norm_r < tol || iter > max_iter)
        {
            // Sum X and return
            X->sum_cols(x);

            delete T;
            delete R;
            delete Z;
            delete P;
            delete P_1;
            delete AP;
            delete AP_1;
            delete X;

            return;
        }

        iter++;

        // Apply preconditioner to AP_k here
        MAP->set_const_value(0.0);
    if (precond_t) *precond_t -= RAPtor_MPI_Wtime();
        ml->cycle(*MAP, *AP);
    if (precond_t) *precond_t += RAPtor_MPI_Wtime();

        // gamma = (AP)^T * (AP)
        AP->mult_T(*MAP, gamma, comp_t);

        // rho = (AP_1)^T * (AP)
        AP_1->mult_T(*MAP, rho, comp_t);

        // Z = AP - P * gamma - P_1 * rho
        AP_1->copy(*AP);
        Z->copy(*MAP);

        P->mult(gamma, *T, comp_t);
        P_1->mult(rho, *AP, comp_t);
        Z->axpy(*T, -1.0, comp_t);
        Z->axpy(*AP, -1.0, comp_t);
       
        // Copy P for next iteration
        P_1->copy(*P);
    }
}
