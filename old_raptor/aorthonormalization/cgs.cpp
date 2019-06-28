// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "aorthonormalization/cgs.hpp"

using namespace raptor;

void BCGS(CSRMatrix* A, BVector& Q, BVector& P)
{
    return;
}

void CGS(CSRMatrix* A, BVector& P)
{
    return;
}

/*void MGS(CSRMatrix* A, aligned_vector<Vector>& W, aligned_vector<aligned_vector<Vector>>& P_list)
{
    Vector Aw(A->n_rows);
    aligned_vector<Vector> Pi;

    int t = W.size();
    int Ps = P_list.size();
    double pAw_inner, wAw_inner;

    // Loop over columns of W
    for (int l = 0; l < t; l++) {
        A->mult(W[l], Aw);
        // Loop over P_i's in P_list
        for (int i = 0; i < Ps; i++) {
            Pi = P_list[i];
            // Loop over columns of P_i
            for (int j = 0; j < t; j++) {
                pAw_inner = Aw.inner_product(Pi[j]);
                W[l].axpy(Pi[j], -1.0 * pAw_inner);
            }
        }
        A->mult(W[l], Aw);
        wAw_inner = Aw.inner_product(W[l]);
        W[l].scale(1/pow(wAw_inner, 2));
    }

    return;
}

void MGS(CSRMatrix* A, aligned_vector<Vector>& P)
{
    Vector Ap(A->n_rows);

    int t = P.size();
    double pAp_inner; 
   
    for (int i = 0; i < t; i++) {
        A->mult(P[i], Ap);
        for (int j = 0; j < i; j++) {
            pAp_inner = Ap.inner_product(P[j]);
            P[i].axpy(P[j], -1.0 * pAp_inner);        
        }
        A->mult(P[i], Ap);
        pAp_inner = Ap.inner_product(P[i]);
        P[i].scale(1/pow(pAp_inner, 2));
    }

    return;
}*/
