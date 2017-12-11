// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_ML_MULTILEVEL_H
#define RAPTOR_ML_MULTILEVEL_H

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "level.hpp"
#include "ruge_stuben/cf_splitting.hpp"
#include "ruge_stuben/interpolation.hpp"
#include "util/linalg/relax.hpp"

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

            Multilevel(CSRMatrix* Af,
                    double strength_threshold = 0.0, // strength threshold
                    coarsen_t coarsen_type = RS, 
                    interp_t interp_type = Direct,
                    relax_t _relax_type = SOR,
                    int _num_smooth_sweeps = 1,
                    double _relax_weight = 1.0,
                    int max_coarse = 50, 
                    int max_levels = -1)
            {
                int last_level = 0;

                levels.push_back(new Level());
                levels[0]->A = new CSRMatrix(Af);
                levels[0]->A->sort();
                levels[0]->x.resize(Af->n_rows);
                levels[0]->b.resize(Af->n_rows);
                levels[0]->tmp.resize(Af->n_rows);
                levels[0]->P = NULL;

                while (levels[last_level]->A->n_rows > max_coarse &&
                        (max_levels == -1 || (int) levels.size() < max_levels))
                {
                    extend_hierarchy(strength_threshold, coarsen_type,
                            interp_type);
                    last_level++;
                }
                num_levels = levels.size();

                form_dense_coarse();

                relax_type = _relax_type;
                relax_weight = _relax_weight;
                num_smooth_sweeps = _num_smooth_sweeps;
            }
 
            ~Multilevel()
            {
                for (std::vector<Level*>::iterator it = levels.begin();
                        it != levels.end(); ++it)
                {
                    delete *it;
                }
            }

            void extend_hierarchy(double strength_threshold,
                    coarsen_t coarsen_type,
                    interp_t interp_type)
            {
                int level_ctr = levels.size() - 1;
                CSRMatrix* A = levels[level_ctr]->A;
                CSRMatrix* S;
                CSRMatrix* P;
                CSRMatrix* AP;
                CSCMatrix* P_csc;
                std::vector<int> states;

                // Form Strength Matrix
                S = A->strength(strength_threshold);
                split_rs(S, states);

                // Form Coarsening (CF Splitting)
                switch (coarsen_type)
                {
                    case RS:
                        split_rs(S, states);
                        break;
                    case CLJP:
                        split_cljp(S, states);
                        break;
                    case Falgout:
                        printf("Falgout in serial is just RS..\n");
                        split_rs(S, states);
                }
                
                // Form interpolation
                switch (interp_type)
                {
                    case Direct:
                        P = direct_interpolation(A, S, states);
                        break;
                    case Classical:
                        P = mod_classical_interpolation(A, S, states);
                        break;
                }
                levels[level_ctr]->P = P;

                // Form coarse-grid operator
                levels.push_back(new Level());
                AP = A->mult(P);
                P_csc = new CSCMatrix(P);
                A = AP->mult_T(P_csc);

                level_ctr++;
                levels[level_ctr]->A = A;
                levels[level_ctr]->x.resize(A->n_rows);
                levels[level_ctr]->b.resize(A->n_rows);
                levels[level_ctr]->tmp.resize(A->n_rows);
                levels[level_ctr]->P = NULL;

                delete AP;
                delete P_csc;
                delete S;
            }

            void form_dense_coarse()
            {
                CSRMatrix* Ac = levels[num_levels-1]->A;
                coarse_n = Ac->n_rows;
                A_coarse.resize(coarse_n*coarse_n, 0);
                for (int i = 0; i < coarse_n; i++)
                {
                    int row_start = Ac->idx1[i];
                    int row_end = Ac->idx1[i+1];
                    for (int j = row_start; j < row_end; j++)
                    {
                        A_coarse[i*coarse_n + Ac->idx2[j]] = Ac->vals[j];
                    }
                }

                LU_permute.resize(coarse_n);
                int info;
                dgetrf_(&coarse_n, &coarse_n, A_coarse.data(), &coarse_n, 
                        LU_permute.data(), &info);
            }

            void cycle(int level)
            {
                CSRMatrix* A = levels[level]->A;
                CSRMatrix* P = levels[level]->P;
                Vector& x = levels[level]->x;
                Vector& b = levels[level]->b;
                Vector& tmp = levels[level]->tmp;


                if (level == num_levels - 1)
                {
                    char trans = 'N'; //No transpose
                    int nhrs = 1; // Number of right hand sides
                    int info; // result
                    double b_data[b.size()];
                    for (int i = 0; i < b.size(); i++)
                        b_data[i] = b.data()[i];
                    dgetrs_(&trans, &coarse_n, &nhrs, A_coarse.data(), &coarse_n, 
                            LU_permute.data(), b_data, &coarse_n, &info);
                    for (int i = 0; i < b.size(); i++)
                        x.data()[i] = b_data[i];
                }
                else
                {
                    // Set next level x to 0.0
                    levels[level+1]->x.set_const_value(0.0);

                    // Relax
                    switch (relax_type)
                    {
                        case Jacobi:
                            jacobi(levels[level], num_smooth_sweeps, relax_weight);
                            break;
                        case SOR:
                            sor(levels[level], num_smooth_sweeps, relax_weight);
                            break;
                        case SSOR:
                            ssor(levels[level], num_smooth_sweeps, relax_weight);
                            break;
                    }

                    // Calculate residual
                    A->residual(x, b, tmp);

                    // Restrict residual
                    P->mult_T(tmp, levels[level+1]->b);

                    // Cycle on coarser levels
                    cycle(level+1);

                    // Interpolate error and add to x
                    P->mult_append(levels[level+1]->x, x);

                    // Relax
                    switch (relax_type)
                    {
                        case Jacobi:
                            jacobi(levels[level], num_smooth_sweeps, relax_weight);
                            break;
                        case SOR:
                            sor(levels[level], num_smooth_sweeps, relax_weight);
                            break;
                        case SSOR:
                            ssor(levels[level], num_smooth_sweeps, relax_weight);
                            break;
                    }
                }
            }

            void solve(Vector& sol, Vector& rhs, int num_iterations = 100)
            {
                double b_norm = rhs.norm(2);
                double r_norm;
                int iter = 0;

                levels[0]->x.copy(sol);
                levels[0]->b.copy(rhs);

                // Iterate until convergence or max iterations
                Vector resid(rhs.size());
                levels[0]->A->residual(levels[0]->x, levels[0]->b, resid);
                if (fabs(b_norm) > zero_tol)
                {
                    r_norm = resid.norm(2) / b_norm;
                }
                else
                {
                    r_norm = resid.norm(2);
                    printf("Small Norm of B -> not using relative residuals\n");
                }
                printf("Rnorm = %e\n", r_norm);

                while (r_norm > 1e-05 && iter < num_iterations)
                {
                    cycle(0);
                    iter++;

                    levels[0]->A->residual(levels[0]->x, levels[0]->b, resid);
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

                sol.copy(levels[0]->x);
            } 

            // Class Parameters to be set before Setup
            relax_t relax_type;
            int num_smooth_sweeps;
            double relax_weight;

            std::vector<Level*> levels;
            std::vector<double> A_coarse;
            std::vector<int> LU_permute;
            int coarse_n;
            int num_levels;

    };
}
#endif
