#ifndef RAPTOR_KRYLOV_PAR_EKCG_HPP
#define RAPTOR_KRYLOV_PAR_EKCG_HPP

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "multilevel/par_multilevel.hpp"
#include <vector>

using namespace raptor;
    
    // BLAS LU routine that is used for coarse solve
    //extern "C" void dgetrs_(char *TRANS, int *N, int *NRHS, double *A, 
    //        int *LDA, int *IPIV, double *B, int *LDB, int *INFO );
extern "C" void dtrsm_(char *SIDE, char *UPLO, char *TRANSA, char *DIAG, int *M, int *N,
        double *ALPHA, double *A, int *LDA, double *B, int *LDB);
extern "C" void dpotrf_(char *UPLO, int *N, double *A, int *LDA, int *INFO );

void EKCG(ParCSRMatrix* A, ParVector& x, ParVector& b, int t, aligned_vector<double>& res, 
        double tol = 1e-05, int max_iter = -1, double* comp_t = NULL, bool tap = false);

#endif
