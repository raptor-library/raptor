// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_MULTILEVEL_H
#define RAPTOR_CORE_MULTILEVEL_H

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "level.hpp"

// Coarse Matrices (A) are CSC
// Prolongation Matrices (P) are CSC
// P^T*A*P is then CSR*(CSC*CSC) -- returns CSC Ac

namespace raptor
{
    // BLAS LU routine that is used for coarse solve
    extern "C" void dgetrf_(int* dim1, int* dim2, double* a, int* lda, 
            int* ipiv, int* info);
    extern "C" void dgetrs_(char *TRANS, int *N, int *NRHS, double *A, 
            int *LDA, int *IPIV, double *B, int *LDB, int *INFO );

    class Multilevel
    {
        public:

            Multilevel(CSRMatrix& Af, double theta = 0.0, double omega = 4.0/3, 
                    int num_smooth_steps = 1, int max_coarse = 50)
            {
                levels.push_back(Af);
                int last_level = 0;
                while (levels[last_level].A.n_rows > max_coarse)
                {
                    levels.push_back(Level(levels[last_level].A,
                                levels[last_level].P,
                                theta,
                                omega,
                                num_smooth_steps,
                                max_coarse));
                    last_level++;
                }
                num_levels = levels.size();

                CSRMatrix& Ac = levels[last_level].A;
                coarse_n = Ac.n_rows;
                A_coarse.resize(coarse_n*coarse_n, 0);
                for (int i = 0; i < coarse_n; i++)
                {
                    int row_start = Ac.idx1[i];
                    int row_end = Ac.idx1[i+1];
                    for (int j = row_start; j < row_end; j++)
                    {
                        A_coarse[i*coarse_n + Ac.idx2[j]] = Ac.vals[j];
                    }
                }

                LU_permute.resize(coarse_n);
                int info;
                dgetrf_(&coarse_n, &coarse_n, A_coarse.data(), &coarse_n, 
                        LU_permute.data(), &info);
            }

            void cycle(int level)
            {
                CSRMatrix& A = levels[level].A;
                CSCMatrix& P = levels[level].P;
                Vector& x = levels[level].x;
                Vector& b = levels[level].b;
                Vector& tmp = levels[level].tmp;


                if (level == num_levels - 1)
                {
                    char trans = 'N'; //No transpose
                    int nhrs = 1; // Number of right hand sides
                    int info; // result
                    double b_data[b.size];
                    for (int i = 0; i < b.size; i++)
                        b_data[i] = b.data()[i];
                    dgetrs_(&trans, &coarse_n, &nhrs, A_coarse.data(), &coarse_n, 
                            LU_permute.data(), b_data, &coarse_n, &info);
                    for (int i = 0; i < b.size; i++)
                        x.data()[i] = b_data[i];
                }
                else
                {
                    levels[level+1].x.set_const_value(0.0);
                    A.gauss_seidel(x, b);
                    A.residual(x, b, tmp);
                    P.mult_T(tmp, levels[level+1].b);
                    cycle(level+1);
                    P.mult_append(levels[level+1].x, x);
                    A.gauss_seidel(x, b);
                }
            }

            void solve(Vector& sol, Vector& rhs, int num_iterations = 100)
            {
                double b_norm = rhs.norm(2);
                double r_norm;
                int iter = 0;

                levels[0].x.copy(&sol);
                levels[0].b.copy(&rhs);

                // Iterate until convergence or max iterations
                Vector resid(rhs.size);
                levels[0].A.residual(levels[0].x, levels[0].b, resid);
                if (fabs(b_norm) > zero_tol)
                {
                    r_norm = resid.norm(2) / b_norm;
                }
                else
                {
                    r_norm = resid.norm(2);
                }
                printf("Rnorm = %e\n", r_norm);

                while (r_norm > 1e-06 && iter < num_iterations)
                {
                    cycle(0);
                    iter++;

                    levels[0].A.residual(levels[0].x, levels[0].b, resid);
                    if (fabs(b_norm) > zero_tol)
                    {
                        r_norm = resid.norm(2) / b_norm;
                    }
                    else
                    {
                        r_norm = resid.norm(2);
                    }
                    printf("Rnorm = %e\n", r_norm);
                }
            } 

            std::vector<Level> levels;
            std::vector<double> A_coarse;
            std::vector<int> LU_permute;
            int coarse_n;
            int num_levels;
    };
}
#endif
