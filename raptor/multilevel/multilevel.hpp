// Copyright (c) 2015-2017, RAPtor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_ML_MULTILEVEL_H
#define RAPTOR_ML_MULTILEVEL_H

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "level.hpp"
#include "ruge_stuben/cf_splitting.hpp"
#include "ruge_stuben/interpolation.hpp"
//#include "smoothed_aggregation/seq/prolongation.hpp"

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

            Multilevel(CSRMatrix* Af, double theta = 0.0, int num_smooth_steps = 1,
                    int max_coarse = 50, int max_levels = -1)
            {
                int last_level = 0;

                levels.push_back(new Level());
                levels[0]->A = new CSRMatrix(Af);
                levels[0]->A->sort();
                levels[0]->x.resize(Af->n_rows);
                levels[0]->b.resize(Af->n_rows);
                levels[0]->tmp.resize(Af->n_rows);

                while (levels[last_level]->A->n_rows > max_coarse &&
                        (max_levels == -1 || (int) levels.size() < max_levels))
                {
                    extend_hierarchy(theta);
                    last_level++;
                }
                num_levels = levels.size();

                form_dense_coarse();
            }

            /*Multilevel(CSRMatrix* Af, data_t* B_ptr = NULL, int num_candidates = 1,
                    double theta = 0.0, double omega = 4.0/3, 
                    int num_smooth_steps = 1, int max_coarse = 50,
                    int max_levels = -1)
            {
                // Always need levels with A, x, b (P and tmp are on all but
                // coarsest)
                levels.push_back(new Level());
                levels[0]->A = new CSRMatrix(Af);
                levels[0]->A->sort();
                levels[0]->x.resize(Af->n_rows);
                levels[0]->b.resize(Af->n_rows);
                levels[0]->tmp.resize(Af->n_rows);

                double* level_B = new data_t[Af->n_rows];
                if (B_ptr)
                {
                    for (int i = 0; i < Af->n_rows; i++)
                    {
                        level_B[i] = B_ptr[i];
                    }
                }
                else
                {
                    for (int i = 0; i < Af->n_rows; i++)
                    {
                        level_B[i] = 1.0;
                    }
                }

                int last_level = 0;
                while (levels[last_level]->A->n_rows > max_coarse &&
                        (max_levels == -1 || levels.size() < max_levels))
                {
                    double* R = extend_hierarchy(level_B, num_candidates,
                            theta, omega);
                    last_level++;
                    delete[] level_B;
                    level_B = R;
                }

                delete[] level_B;

                num_levels = levels.size();

                form_dense_coarse();
            }*/

            ~Multilevel()
            {
                for (std::vector<Level*>::iterator it = levels.begin();
                        it != levels.end(); ++it)
                {
                    delete *it;
                }
            }

            void extend_hierarchy(double theta = 0.0)
            {
                int level_ctr = levels.size() - 1;
                CSRMatrix* A = levels[level_ctr]->A;
                CSRMatrix* S;
                CSRMatrix* P;
                CSRMatrix* AP;
                CSCMatrix* P_csc;
                std::vector<int> states;

                S = A->strength(theta);
                split_rs(S, states);
                levels[level_ctr]->P = direct_interpolation(A, S, states);
                P = levels[level_ctr]->P;

                levels.push_back(new Level());
                AP = A->mult(P);
                P_csc = new CSCMatrix(P);

                level_ctr++;
                levels[level_ctr]->A = AP->mult_T(P_csc);
                A = levels[level_ctr]->A;
                levels[level_ctr]->x.resize(A->n_rows);
                levels[level_ctr]->b.resize(A->n_rows);
                levels[level_ctr]->tmp.resize(A->n_rows);
                levels[level_ctr]->P = NULL;

                delete AP;
                delete P_csc;
                delete S;
            }

            /*data_t* extend_hierarchy(data_t* B,
                    int num_candidates = 1,
                    double theta = 0.0, 
                    double omega = 4.0/3)
            {
                int level_ctr = levels.size()-1;

                CSRMatrix* S;
                CSRMatrix* AggOp;
                CSRMatrix* T;
                CSRMatrix* AP;
                CSCMatrix* P_csc;

                // Create Strength of Connection
                S = levels[level_ctr]->A->strength(theta);

                // Use standard aggregation
                AggOp = S->aggregate();

                // Create tentative interpolation
                data_t* R = new data_t[AggOp->n_cols];
                T = AggOp->fit_candidates(B, R, num_candidates);

                // Smooth T to form prolongation
                levels[level_ctr]->P = jacobi_prolongation(levels[level_ctr]->A, T, 
                        omega, num_smooth_steps);

                // Create coarse A
                levels.push_back(new Level());
                AP = (levels[level_ctr]->A)->mult(levels[level_ctr]->P);
                P_csc = new CSCMatrix(levels[level_ctr]->P);
                levels[level_ctr+1]->A = AP->mult_T(P_csc);
                level_ctr++;

        	    // Sort coarse A
		        levels[level_ctr]->A->sort();

                // Resize vectors to equal shape of A
                levels[level_ctr]->x.resize(levels[level_ctr]->A->n_rows);
                levels[level_ctr]->b.resize(levels[level_ctr]->A->n_rows);
                levels[level_ctr]->tmp.resize(levels[level_ctr]->A->n_rows);
                
                return R;
            }*/

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
                    levels[level+1]->x.set_const_value(0.0);
                    A->gauss_seidel(x, b);
                    A->residual(x, b, tmp);
                    P->mult_T(tmp, levels[level+1]->b);
                    cycle(level+1);
                    P->mult_append(levels[level+1]->x, x);
                    A->gauss_seidel(x, b);
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

            std::vector<Level*> levels;
            std::vector<double> A_coarse;
            std::vector<int> LU_permute;
            int coarse_n;
            int num_levels;
    };
}
#endif
