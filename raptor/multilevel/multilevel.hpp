// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_MULTILEVEL_H
#define RAPTOR_CORE_MULTILEVEL_H

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "level.hpp"
#include "aggregation/prolongation.hpp"

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

            Multilevel(ParCSRMatrix* Af, data_t* B_ptr = NULL, int num_candidates = 1,
                    double theta = 0.0, double omega = 4.0/3, 
                    int num_smooth_steps = 1, int max_coarse = 50)
            {
                // Always need levels with A, x, b (P and tmp are on all but
                // coarsest)
                levels.push_back(new Level());
                levels[0]->A = new ParCSRMatrix(Af);
                levels[0]->A->sort();
                levels[0]->x.set_size(Af->global_num_rows, Af->local_num_rows,
                        Af->first_local_row);
                levels[0]->b.set_size(Af->global_num_rows, Af->local_num_rows,
                        Af->first_local_row);
                levels[0]->tmp.set_size(Af->global_num_rows, Af->local_num_rows,
                        Af->first_local_row);

                double* level_B = new data_t[Af->local_num_rows];
                if (B_ptr)
                {
                    for (int i = 0; i < Af->local_num_rows; i++)
                    {
                        level_B[i] = B_ptr[i];
                    }
                }
                else
                {
                    for (int i = 0; i < Af->local_num_rows; i++)
                    {
                        level_B[i] = 1.0;
                    }
                }


                int last_level = 0;
                while (levels[last_level]->A->global_num_rows > max_coarse)
                {
                    double* R = extend_hierarchy(level_B, num_candidates,
                            theta, omega, num_smooth_steps, max_coarse);
                    last_level++;
                    delete[] level_B;
                    level_B = R;
                }

                delete[] level_B;

                num_levels = levels.size();

                // TODO -- gather Ac so that each process with any local num
                // rows has all of Ac stored locally (in a dense matrix)
                ParCSRMatrix* Ac = levels[last_level]->A;
                coarse_n = Ac->global_num_rows;
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

                printf("Num Levels = %d\n", num_levels);
            }

            ~Multilevel()
            {
                for (std::vector<Level*>::iterator it = levels.begin();
                        it != levels.end(); ++it)
                {
                    delete *it;
                }
            }

            data_t* extend_hierarchy(data_t* B,
                    int num_candidates = 1,
                    double theta = 0.0, 
                    double omega = 4.0/3, 
                    int num_smooth_steps = 1,
                    int max_coarse = 50)
            {
                int level_ctr = levels.size()-1;

                ParCSRMatrix* S;
                ParCSRMatrix* AggOp;
                ParCSRMatrix* T;
                ParCSRMatrix* AP;
                ParCSCMatrix* P_csc;

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
                levels[level_ctr]->x.set_size(levels[level_ctr]->A->n_rows);
                levels[level_ctr]->b.set_size(levels[level_ctr]->A->n_rows);
                levels[level_ctr]->tmp.set_size(levels[level_ctr]->A->n_rows);
                
                return R;
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
                Vector resid(rhs.size);
                levels[0]->A->residual(levels[0]->x, levels[0]->b, resid);
//                if (fabs(b_norm) > zero_tol)
                {
//                    r_norm = resid.norm(2) / b_norm;
                }
//                else
                {
                    r_norm = resid.norm(2);
                }
                printf("Rnorm = %e\n", r_norm);

                while (r_norm > 1e-05 && iter < num_iterations)
                {
                    cycle(0);
                    iter++;

                    levels[0]->A->residual(levels[0]->x, levels[0]->b, resid);
//                    if (fabs(b_norm) > zero_tol)
                    {
//                        r_norm = resid.norm(2) / b_norm;
                    }
//                    else
                    {
                        r_norm = resid.norm(2);
                    }
                    printf("Rnorm = %e\n", r_norm);
                }
            } 

            std::vector<Level*> levels;
            std::vector<double> A_coarse;
            std::vector<int> LU_permute;
            int coarse_n;
            int num_levels;
    };
}
#endif
