// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_ML_PARMULTILEVEL_H
#define RAPTOR_ML_PARMULTILEVEL_H

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "multilevel/par_level.hpp"
#include "util/linalg/par_relax.hpp"
#include "ruge_stuben/par_interpolation.hpp"
#include "ruge_stuben/par_cf_splitting.hpp"
#include "multilevel/par_sparsify.hpp"

#ifdef USING_HYPRE
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"
#endif

/**************************************************************
 *****   ParMultilevel Class
 **************************************************************
 ***** This class constructs a parallel multigrid hierarchy
 *****
 ***** Attributes
 ***** -------------
 ***** Af : ParCSRMatrix*
 *****    Fine-level matrix
 ***** strength_threshold : double (default 0.0)
 *****    Threshold for strong connections
 ***** coarsen_type : coarsen_t (default Falgout)
 *****    Type of coarsening scheme.  Options are 
 *****      - RS : ruge stuben splitting
 *****      - CLJP 
 *****      - Falgout : RS on_proc, but CLJP on processor boundaries
 *****      - PMIS 
 *****      - HMIS
 ***** interp_type : interp_t (default Direct)
 *****    Type of interpolation scheme.  Options are
 *****      - Direct 
 *****      - Classical (modified classical interpolation)
 *****      - Extended (extended + i interpolation)
 ***** relax_type : relax_t (default SOR)
 *****    Relaxation scheme used in every cycle of solve phase.
 *****    Options are:
 *****      - Jacobi: weighted jacobi for both on and off proc
 *****      - SOR: weighted jacobi off_proc, SOR on_proc
 *****      - SSOR : weighted jacobi off_proc, SSOR on_proc
 ***** num_smooth_sweeps : int (defualt 1)
 *****    Number of relaxation sweeps (both pre and post smoothing)
 *****    to be performed during each cycle of the AMG solve.
 ***** relax_weight : double
 *****    Weight used in Jacobi, SOR, or SSOR
 ***** max_coarse : int (default 50)
 *****    Maximum global num rows allowed in coarsest matrix
 ***** max_levels : int (default -1)
 *****    Maximum number of levels in hierarchy, or no maximum if -1
 ***** 
 ***** Methods
 ***** -------
 ***** solve(x, b, num_iters)
 *****    Solves system Ax = b, performing at most num_iters iterations
 *****    of AMG.
 **************************************************************/

namespace raptor
{
    // BLAS LU routine that is used for coarse solve
    extern "C" void dgetrf_(int* dim1, int* dim2, double* a, int* lda, 
            int* ipiv, int* info);
    extern "C" void dgetrs_(char *TRANS, int *N, int *NRHS, double *A, 
            int *LDA, int *IPIV, double *B, int *LDB, int *INFO );

    class ParMultilevel
    {
        public:

            ParMultilevel(double _strong_threshold = 0.0, 
                    coarsen_t _coarsen_type = Falgout, 
                    interp_t _interp_type = Direct,
                    relax_t _relax_type = SOR) // which level to start tap_amg (-1 == no TAP)
            {
                strong_threshold = _strong_threshold;
                coarsen_type = _coarsen_type;
                interp_type = _interp_type;
                relax_type = _relax_type;
                num_smooth_sweeps = 1;
                relax_weight = 1.0;
                max_coarse = 50;
                max_levels = 25;
                num_variables = 1;
                tap_amg = -1;
                weights = NULL;
                variables = NULL;
                store_residuals = true;
                sparsify_tol = 0.0;
            }

            ~ParMultilevel()
            {
                if (levels[num_levels-1]->A->local_num_rows)
                {
                    MPI_Comm_free(&coarse_comm);
                }
                for (std::vector<ParLevel*>::iterator it = levels.begin();
                        it != levels.end(); ++it)
                {
                    delete *it;
                }
            }

            void setup(ParCSRMatrix* Af)
            {
                double t0;
                int rank, num_procs;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
                int last_level = 0;

                t0 = MPI_Wtime();

                // Add original, fine level to hierarchy
                levels.push_back(new ParLevel());
                levels[0]->A = new ParCSRMatrix(Af);
                levels[0]->A->sort();
                levels[0]->A->on_proc->move_diag();
                levels[0]->x.resize(Af->global_num_rows, Af->local_num_rows,
                        Af->partition->first_local_row);
                levels[0]->b.resize(Af->global_num_rows, Af->local_num_rows,
                        Af->partition->first_local_row);
                levels[0]->tmp.resize(Af->global_num_rows, Af->local_num_rows,
                        Af->partition->first_local_row);
                if (tap_amg == 0 && !Af->tap_comm)
                {
                    levels[0]->A->tap_comm = new TAPComm(Af->partition,
                            Af->off_proc_column_map, Af->on_proc_column_map);
                }
                strength_times.push_back(0);
                coarsen_times.push_back(0);
                interp_times.push_back(0);
                matmat_times.push_back(0);
                matmat_comm_times.push_back(0);

                if (weights == NULL)
                {
                    form_rand_weights(Af->local_num_rows, Af->partition->first_local_row);
                }

                if (num_variables > 1 && variables == NULL) 
                {
                    form_variable_list(Af, num_variables);
                }

                // Add coarse levels to hierarchy 
                while (levels[last_level]->A->global_num_rows > max_coarse && 
                        (max_levels == -1 || (int) levels.size() < max_levels))
                {
                    extend_hierarchy();
                    last_level++;

                    strength_times.push_back(0);
                    coarsen_times.push_back(0);
                    interp_times.push_back(0);
                    matmat_times.push_back(0);
                    matmat_comm_times.push_back(0);
                }

                if (sparsify_tol > 0.0)
                {
                    for (int i = 0; i < num_levels-1; i++)
                    {
                        ParLevel* l = levels[i];
                        sparsify(l->A, l->P, l->I, l->AP, levels[i+1]->A, sparsify_tol);
                        delete l->AP;
                        delete l->I;
                        l->AP = NULL;
                        l->I = NULL;
                    }
                }

                num_levels = levels.size();
                delete[] variables;
                delete[] weights;

                // Duplicate coarsest level across all processes that hold any
                // rows of A_c
                duplicate_coarse();

                setup_times.push_back(MPI_Wtime() - t0);
            }

            void form_variable_list(const ParCSRMatrix* A, const int num_variables)
            {
                if (A->local_num_rows == 0 || num_variables <= 1) return;
                
                variables = new int[A->local_num_rows];
                int var_dist = A->global_num_rows / num_variables;
                for (int i = 0; i < A->local_num_rows; i++)
                {
                    variables[i] = A->local_row_map[i] % num_variables;
                }
            }

            void form_rand_weights(int local_n, int first_n)
            {
                if (local_n == 0) return;

                weights = new double[local_n];
                srand(2448422 + first_n);
                for (int i = 0; i < local_n; i++)
                {
                    weights[i] = double(rand())/RAND_MAX;
                }
            }
                
            void extend_hierarchy()
            {
                int level_ctr = levels.size() - 1;
                bool tap_level = tap_amg >= 0 && tap_amg <= level_ctr;

                ParCSRMatrix* A = levels[level_ctr]->A;
                ParCSRMatrix* S;
                ParCSRMatrix* P;
                ParCSRMatrix* AP;
                ParCSCMatrix* P_csc;

                std::vector<int> states;
                std::vector<int> off_proc_states;

                // Form strength of connection
                strength_times[level_ctr] -= MPI_Wtime();
                S = A->strength(strong_threshold, num_variables, variables);
                strength_times[level_ctr] += MPI_Wtime();

                // Form CF Splitting
                coarsen_times[level_ctr] -= MPI_Wtime();
                switch (coarsen_type)
                {
                    case RS:
                        if (level_ctr < 3) split_rs(S, states, off_proc_states, tap_level);
                        else split_falgout(S, states, off_proc_states, tap_level);
                        break;
                    case CLJP:
                        split_cljp(S, states, off_proc_states, tap_level, weights);
                        break;
                    case Falgout:
                        split_falgout(S, states, off_proc_states, tap_level, weights);
                        break;
                    case PMIS:
                        split_pmis(S, states, off_proc_states, tap_level, weights);
                        break;
                    case HMIS:
			            split_hmis(S, states, off_proc_states, tap_level, weights);
                        break;
                }
                coarsen_times[level_ctr] += MPI_Wtime();

                // Form modified classical interpolation
                interp_times[level_ctr] -= MPI_Wtime();
                switch (interp_type)
                {
                    case Direct:
                        P = direct_interpolation(A, S, states, off_proc_states);
                        break;
                    case Classical:
                        P = mod_classical_interpolation(A, S, states, off_proc_states, 
                                false, num_variables, variables);
                        break;
                    case Extended:
                        P = extended_interpolation(A, S, states, off_proc_states, tap_level, num_variables, variables);
                        break;
                }
                interp_times[level_ctr] += MPI_Wtime();
                levels[level_ctr]->P = P;

                if (num_variables > 1)
                {
                    int ctr = 0;
                    for (int i = 0; i < A->local_num_rows; i++)
                    {
                        if (states[i] == 1)
                        {
                            variables[ctr++] = variables[i];
                        }
                    }
                }

                // Form coarse grid operator
                levels.push_back(new ParLevel());

                if (tap_level)
                {
                    AP = A->tap_mult(levels[level_ctr]->P);
                    P_csc = new ParCSCMatrix(levels[level_ctr]->P);
                    A = AP->tap_mult_T(P_csc);
                }
                else
                {
                    AP = A->mult(levels[level_ctr]->P);
                    P_csc = new ParCSCMatrix(levels[level_ctr]->P);
                    A = AP->mult_T(P_csc);
                }

                level_ctr++;
                levels[level_ctr]->A = A;
                A->comm = new ParComm(A->partition, A->off_proc_column_map,
                        A->on_proc_column_map);
                levels[level_ctr]->x.resize(A->global_num_rows, A->local_num_rows,
                        A->partition->first_local_row);
                levels[level_ctr]->b.resize(A->global_num_rows, A->local_num_rows,
                        A->partition->first_local_row);
                levels[level_ctr]->tmp.resize(A->global_num_rows, A->local_num_rows,
                        A->partition->first_local_row);
                levels[level_ctr]->P = NULL;

                if (tap_amg >= 0 && tap_amg <= level_ctr)
                {
                    levels[level_ctr]->A->tap_comm = new TAPComm(
                            levels[level_ctr]->A->partition,
                            levels[level_ctr]->A->off_proc_column_map,
                            levels[level_ctr]->A->on_proc_column_map);
                }

                if (sparsify_tol > 0.0)
                {
                    levels[level_ctr-1]->AP = AP;
                    
                    // Create and store injection
                    ParCSRMatrix* I = new ParCSRMatrix(P->partition, P->global_num_rows,
                            P->global_num_cols, P->local_num_rows, P->on_proc_num_cols, 0);

                    I->on_proc->idx1[0] = 0;
                    I->off_proc->idx1[0] = 0;
                    int ctr = 0;
                    for (int i = 0; i < A->local_num_rows; i++)
                    {
                        if (states[i])
                        {
                            I->on_proc->idx2.push_back(ctr++);
                            I->on_proc->vals.push_back(1.0);
                        }
                        I->on_proc->idx1[i+1] = I->on_proc->idx2.size();
                        I->off_proc->idx1[i+1] = I->off_proc->idx2.size();
                    }
                    I->on_proc->nnz = I->on_proc->idx2.size();
                    I->off_proc->nnz = I->off_proc->idx2.size();
                    I->finalize();
                    levels[level_ctr-1]->I = I;
                }
                else
                {
                    delete AP;
                }

                delete P_csc;
                delete S;
            }

             void duplicate_coarse()
            {
                int rank, num_procs;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

                int last_level = num_levels - 1;
                ParCSRMatrix* Ac = levels[last_level]->A;
                std::vector<int> proc_sizes(num_procs);
                std::vector<int> active_procs;
                MPI_Allgather(&(Ac->local_num_rows), 1, MPI_INT, proc_sizes.data(),
                        1, MPI_INT, MPI_COMM_WORLD);
                for (int i = 0; i < num_procs; i++)
                {
                    if (proc_sizes[i])
                    {
                        active_procs.push_back(i);
                    }
                }
                MPI_Group world_group;
                MPI_Comm_group(MPI_COMM_WORLD, &world_group);                
                MPI_Group active_group;
                MPI_Group_incl(world_group, active_procs.size(), active_procs.data(),
                        &active_group);
                MPI_Comm_create_group(MPI_COMM_WORLD, active_group, 0, &coarse_comm);
                if (Ac->local_num_rows)
                {
                    int num_active, active_rank;
                    MPI_Comm_rank(coarse_comm, &active_rank);
                    MPI_Comm_size(coarse_comm, &num_active);

                    int proc;
                    int global_col, local_col;
                    int start, end;

                    std::vector<double> A_coarse_lcl;

                    // Gather global col indices
                    coarse_sizes.resize(num_active);
                    coarse_displs.resize(num_active+1);
                    coarse_displs[0] = 0;
                    for (int i = 0; i < num_active; i++)
                    {
                        proc = active_procs[i];
                        coarse_sizes[i] = proc_sizes[proc];
                        coarse_displs[i+1] = coarse_displs[i] + coarse_sizes[i]; 
                    }

                    std::vector<int> global_row_indices(coarse_displs[num_active]);

                    MPI_Allgatherv(Ac->local_row_map.data(), Ac->local_num_rows, MPI_INT,
                            global_row_indices.data(), coarse_sizes.data(), 
                            coarse_displs.data(), MPI_INT, coarse_comm);
    
                    std::map<int, int> global_to_local;
                    int ctr = 0;
                    for (std::vector<int>::iterator it = global_row_indices.begin();
                            it != global_row_indices.end(); ++it)
                    {
                        global_to_local[*it] = ctr++;
                    }

                    coarse_n = Ac->global_num_rows;
                    A_coarse_lcl.resize(coarse_n*Ac->local_num_rows, 0);
                    for (int i = 0; i < Ac->local_num_rows; i++)
                    {
                        start = Ac->on_proc->idx1[i];
                        end = Ac->on_proc->idx1[i+1];
                        for (int j = start; j < end; j++)
                        {
                            global_col = Ac->on_proc_column_map[Ac->on_proc->idx2[j]];
                            local_col = global_to_local[global_col];
                            A_coarse_lcl[i*coarse_n + local_col] = Ac->on_proc->vals[j];
                        }

                        start = Ac->off_proc->idx1[i];
                        end = Ac->off_proc->idx1[i+1];
                        for (int j = start; j < end; j++)
                        {
                            global_col = Ac->off_proc_column_map[Ac->off_proc->idx2[j]];
                            local_col = global_to_local[global_col];
                            A_coarse_lcl[i*coarse_n + local_col] = Ac->off_proc->vals[j];
                        }
                    }

                    A_coarse.resize(coarse_n*coarse_n);
                    for (int i = 0; i < num_active; i++)
                    {
                        coarse_sizes[i] *= coarse_n;
                        coarse_displs[i+1] *= coarse_n;
                    }
                    
                    MPI_Allgatherv(A_coarse_lcl.data(), A_coarse_lcl.size(), MPI_DOUBLE,
                            A_coarse.data(), coarse_sizes.data(), coarse_displs.data(), 
                            MPI_DOUBLE, coarse_comm);

                    LU_permute.resize(coarse_n);
                    int info;
                    dgetrf_(&coarse_n, &coarse_n, A_coarse.data(), &coarse_n, 
                            LU_permute.data(), &info);

                    for (int i = 0; i < num_active; i++)
                    {
                        coarse_sizes[i] /= coarse_n;
                        coarse_displs[i+1] /= coarse_n;
                    }
                }
            }

            void cycle(ParVector& x, ParVector& b, int level = 0, int tap_level = -1)
            {
                ParCSRMatrix* A = levels[level]->A;
                ParCSRMatrix* P = levels[level]->P;
                ParVector& tmp = levels[level]->tmp;

                if (level == num_levels - 1)
                {
                    if (A->local_num_rows)
                    {
                        int active_rank;
                        MPI_Comm_rank(coarse_comm, &active_rank);

                        char trans = 'N'; //No transpose
                        int nhrs = 1; // Number of right hand sides
                        int info; // result

                        std::vector<double> b_data(coarse_n);
                        MPI_Allgatherv(b.local.data(), b.local_n, MPI_DOUBLE, b_data.data(), 
                                coarse_sizes.data(), coarse_displs.data(), 
                                MPI_DOUBLE, coarse_comm);

                        dgetrs_(&trans, &coarse_n, &nhrs, A_coarse.data(), &coarse_n, 
                                LU_permute.data(), b_data.data(), &coarse_n, &info);
                        for (int i = 0; i < b.local_n; i++)
                        {
                            x.local[i] = b_data[i + coarse_displs[active_rank]];
                        }
                    }
                }
                else if (tap_level >= 0 && tap_level <= level) // TAP AMG
                {
                    levels[level+1]->x.set_const_value(0.0);
                    
                    // Relax
                    switch (relax_type)
                    {
                        case Jacobi:
                            tap_jacobi(A, x, b, tmp, num_smooth_sweeps, relax_weight);
                            break;
                        case SOR:
                            tap_sor(A, x, b, tmp, num_smooth_sweeps, relax_weight);
                            break;
                        case SSOR:
                            tap_ssor(A, x, b, tmp, num_smooth_sweeps, relax_weight);
                            break;
                    }


                    A->tap_residual(x, b, tmp);
                    P->tap_mult_T(tmp, levels[level+1]->b);

                    cycle(levels[level+1]->x, levels[level+1]->b, level+1, tap_level);

                    P->tap_mult(levels[level+1]->x, tmp);
                    for (int i = 0; i < A->local_num_rows; i++)
                    {
                        x.local[i] += tmp.local[i];
                    }

                    switch (relax_type)
                    {
                        case Jacobi:
                            tap_jacobi(A, x, b, tmp, num_smooth_sweeps, relax_weight);
                            break;
                        case SOR:
                            tap_sor(A, x, b, tmp, num_smooth_sweeps, relax_weight);
                            break;
                        case SSOR:
                            tap_ssor(A, x, b, tmp, num_smooth_sweeps, relax_weight);
                            break;
                    }

                }
                else // Standard AMG
                {
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

                    A->residual(x, b, tmp);
                    P->mult_T(tmp, levels[level+1]->b);

                    cycle(levels[level+1]->x, levels[level+1]->b, level+1, tap_level);

                    P->mult(levels[level+1]->x, tmp);
                    for (int i = 0; i < A->local_num_rows; i++)
                    {
                        x.local[i] += tmp.local[i];
                    }

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

            int tap_solve(ParVector& sol, ParVector& rhs, int tap_level = 0, int num_iterations = 100)
            {
                double b_norm = rhs.norm(2);
                double r_norm;
                int iter = 0;

                if (store_residuals)
                {
                    residuals.resize(num_iterations + 1);
                }

                // Iterate until convergence or max iterations
                ParVector resid(rhs.global_n, rhs.local_n, rhs.first_local);
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

                while (r_norm > 1e-07 && iter < num_iterations)
                {
                    cycle(sol, rhs, 0, tap_level);

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

            int solve(ParVector& sol, ParVector& rhs, int num_iterations = 100)
            {
                return tap_solve(sol, rhs, -1, num_iterations);
            } 

            std::vector<double>& get_residuals()
            {
                return residuals;
            }

            coarsen_t coarsen_type;
            interp_t interp_type;
            relax_t relax_type;

            int num_smooth_sweeps;
            int max_coarse;
            int max_levels;
            int tap_amg;
            int num_variables;

            double strong_threshold;
            double relax_weight;
            double sparsify_tol;

            bool store_residuals;

            int* variables;
            double* weights;
            std::vector<double> residuals;

            std::vector<ParLevel*> levels;
            std::vector<int> LU_permute;
            int num_levels;
            
            std::vector<double> spmv_times;
            std::vector<double> spmv_comm_times;
            std::vector<double> setup_times;
            std::vector<double> strength_times;
            std::vector<double> coarsen_times;
            std::vector<double> interp_times;
            std::vector<double> matmat_times;
            std::vector<double> matmat_comm_times;
            
            double setup_comm_t;
            double solve_comm_t;
            int setup_comm_n;
            int setup_comm_s;
            int solve_comm_n;
            int solve_comm_s;

            int coarse_n;
            std::vector<double> A_coarse;
            std::vector<int> coarse_sizes;
            std::vector<int> coarse_displs;
            MPI_Comm coarse_comm;
    };
}
#endif
