// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_ML_MULTILEVEL_H
#define RAPTOR_ML_MULTILEVEL_H

#include "core/types.hpp"
#include "core/matrix.hpp"
#ifndef NO_CUDA
    #include "core/cuda/vector_cuda.hpp"
#else
    #include "core/serial/vector.hpp"
#endif
#include "level.hpp"
#include "relaxation/relax.hpp"

// Coarse Matrices (A) are CSC
// Prolongation Matrices (P) are CSC
// P^T*A*P is then CSR*(CSC*CSC) -- returns CSC Ac

/**************************************************************
 *****   Multilevel Base Class
 **************************************************************
 ***** This class constructs a multilevel object, outlining
 ***** the AMG structure
 **************************************************************/
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

            Multilevel(double _strong_threshold,
                    strength_t _strength_type,
                    relax_t _relax_type)
            {
                strong_threshold = _strong_threshold;
                strength_type = _strength_type,
                relax_type = _relax_type;
                num_smooth_sweeps = 1;
                relax_weight = 1.0;
                max_coarse = 50;
                max_levels = 25;
                weights = NULL;
                store_residuals = true;
            }

            virtual ~Multilevel()
            {
                for (std::vector<Level*>::iterator it = levels.begin();
                        it != levels.end(); ++it)
                {
                    delete *it;
                }
            }

            virtual void setup(CSRMatrix* Af) = 0;

            void setup_helper(CSRMatrix* Af)
            {
                printf("Strength %d\n", strength_type);
                int last_level = 0;

                if (weights == NULL)
                {
                    form_rand_weights(Af->n_rows);
                }

                levels.emplace_back(new Level());
                levels[0]->A = Af->copy();
                levels[0]->A->sort();
                levels[0]->x.resize(Af->n_rows);
                levels[0]->b.resize(Af->n_rows);
                levels[0]->tmp.resize(Af->n_rows);
                levels[0]->P = NULL;

                while (levels[last_level]->A->n_rows > max_coarse &&
                        (max_levels == -1 || (int) levels.size() < max_levels))
                {
                    extend_hierarchy();
                    last_level++;
                }
                num_levels = levels.size();

                delete[] weights;
                weights = NULL;

                form_dense_coarse();
            }

            void form_rand_weights(int n)
            {
                if (n == 0) return;

                weights = new double[n];
                srand(2448422);
                for (int i = 0; i < n; i++)
                {
                    weights[i] = double(rand()) / RAND_MAX;
                }
            }
 
            virtual void extend_hierarchy() = 0;

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

            void cycle(Vector& x, Vector& b, int level)
            {
                CSRMatrix* A = levels[level]->A;
                CSRMatrix* P = levels[level]->P;
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
                            jacobi(A, x, b, tmp, num_smooth_sweeps, relax_weight);
                            break;
                        case SOR:
                            sor(A, x, b, tmp, num_smooth_sweeps, relax_weight);
                            break;
                        case SSOR:
                            ssor(A, x, b, tmp, num_smooth_sweeps, relax_weight);
                            break;
                    }

                    // Calculate residual
                    A->residual(x, b, tmp);

                    // Restrict residual
                    P->mult_T(tmp, levels[level+1]->b);

                    // Cycle on coarser levels
                    cycle(levels[level+1]->x, levels[level+1]->b, level+1);

                    // Interpolate error and add to x
                    P->mult_append(levels[level+1]->x, x);

                    // Relax
                    switch (relax_type)
                    {
                        case Jacobi:
                            jacobi(A, x, b, tmp, num_smooth_sweeps, relax_weight);
                            break;
                        case SOR:
                            sor(A, x, b, tmp, num_smooth_sweeps, relax_weight);
                            break;
                        case SSOR:
                            ssor(A, x, b, tmp, num_smooth_sweeps, relax_weight);
                            break;
                    }
                }
            }

            int solve(Vector& sol, Vector& rhs, int num_iterations = 100)
            {
                double b_norm = rhs.norm(2);
                double r_norm;
                int iter = 0;

                if (store_residuals)
                {
                    residuals.resize(num_iterations + 1);
                }

                // Iterate until convergence or max iterations
                Vector resid(rhs.size());
                levels[0]->A->residual(sol, rhs, resid);
                if (fabs(b_norm) > zero_tol)
                {
                    r_norm = resid.norm(2) / b_norm;
                }
                else
                {
                    r_norm = resid.norm(2);
                    printf("Small Norm of B -> not using relative residuals\n");
                }

                if (store_residuals)
                {
                    residuals[iter] = r_norm;
                }

                while (r_norm > 1e-07 && iter < num_iterations)
                {
                    cycle(sol, rhs, 0);

                    iter++;
                    levels[0]->A->residual(sol, rhs, resid);
                    if (fabs(b_norm) > zero_tol)
                    {
                        r_norm = resid.norm(2) / b_norm;
                    }
                    else
                    {
                        r_norm = resid.norm(2);
                    }
                    if (store_residuals)
                    {
                        residuals[iter] = r_norm;
                    }
                }
                return iter;
            } 

            aligned_vector<double>& get_residuals()
            {
                return residuals;
            }

            relax_t relax_type;
            strength_t strength_type;

            int num_smooth_sweeps;
            int max_coarse;
            int max_levels;

            double strong_threshold;
            double relax_weight;

            bool store_residuals;

            double* weights;
            aligned_vector<double> residuals;

            std::vector<Level*> levels;
            aligned_vector<double> A_coarse;
            aligned_vector<int> LU_permute;
            int coarse_n;
            int num_levels;

    };
}
#endif
