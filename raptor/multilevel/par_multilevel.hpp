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

            ParMultilevel(double _strong_threshold, 
                    strength_t _strength_type,
                    relax_t _relax_type) // which level to start tap_amg (-1 == no TAP)
            {
                num_levels = 0;
                strong_threshold = _strong_threshold;
                strength_type = _strength_type;
                relax_type = _relax_type;
                num_smooth_sweeps = 1;
                relax_weight = 1.0;
                max_coarse = 50;
                max_levels = 25;
                tap_amg = -1;
                tap_simple = false;
                weights = NULL;
                store_residuals = true;
                track_times = false;
                setup_times = NULL;
                solve_times = NULL;
                sparsify_tol = 0.0;
                solve_tol = 1e-07;
                max_iterations = 100;
            }

            virtual ~ParMultilevel()
            {
                if (num_levels > 0)
                {
                    if (levels[num_levels-1]->A->local_num_rows)
                    {
                        RAPtor_MPI_Comm_free(&coarse_comm);
                    }
                }

                for (std::vector<ParLevel*>::iterator it = levels.begin();
                        it != levels.end(); ++it)
                {
                    delete *it;
                }

                delete[] weights;

                delete[] setup_times;
                delete[] solve_times;
            }
            
            virtual void setup(ParCSRMatrix* Af, int nrhs = 1) = 0;

            void setup_helper(ParCSRMatrix* Af, int nrhs = 1)
            {
                int rank, num_procs;
                RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
                RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);
                int last_level = 0;

                if (track_times)
                {
                    setup_times = new double[5 * max_levels]();
                    init_profile();
                }

                // Add original, fine level to hierarchy
                levels.emplace_back(new ParLevel());
                levels[0]->A = Af->copy();
                levels[0]->A->sort();
                levels[0]->A->on_proc->move_diag();
                levels[0]->x.local->b_vecs = nrhs;
                levels[0]->x.resize(Af->global_num_rows, Af->local_num_rows);
                levels[0]->b.local->b_vecs = nrhs;
                levels[0]->b.resize(Af->global_num_rows, Af->local_num_rows);
                levels[0]->tmp.local->b_vecs = nrhs;
                levels[0]->tmp.resize(Af->global_num_rows, Af->local_num_rows);
                if (tap_amg == 0)
                {
                    if (!Af->tap_comm && !Af->tap_mat_comm)
                    {
                        levels[0]->A->init_tap_communicators();
                    }
                    else if (!Af->tap_comm) // 3-step NAPComm
                    {
                        levels[0]->A->tap_comm = new TAPComm(Af->partition,
                                Af->off_proc_column_map, Af->on_proc_column_map);
                    }
                    else if (!Af->tap_mat_comm) // 2-step NAPComm
                    {
                        levels[0]->A->tap_mat_comm = new TAPComm(Af->partition,
                                Af->off_proc_column_map, Af->on_proc_column_map, false);
                    }
                }

                if (weights == NULL)
                {
                    form_rand_weights(Af->local_num_rows, Af->partition->first_local_row);
                }

                // Add coarse levels to hierarchy 
                while (levels[last_level]->A->global_num_rows > max_coarse && 
                        (max_levels == -1 || (int) levels.size() < max_levels))
                {
                    extend_hierarchy(nrhs);

                    if (track_times)
                    {
                        finalize_profile();
                        setup_times[5*last_level] = total_t;
                        setup_times[5*last_level + 1] = collective_t;
                        setup_times[5*last_level + 2] = p2p_t;
                        setup_times[5*last_level + 3] = vec_t;
                        setup_times[5*last_level + 4] = mat_t;
                        init_profile();
                    }

                    last_level++;
                }

                num_levels = levels.size();
                if (Af->local_num_rows) 
                {
                    delete[] weights;
                    weights = NULL;
                }

                // Duplicate coarsest level across all processes that hold any
                // rows of A_c
                duplicate_coarse();

                if (track_times)
                {
                    finalize_profile();
                    setup_times[5*(num_levels-1)] += total_t;
                    setup_times[5*(num_levels-1) + 1] += collective_t;
                    setup_times[5*(num_levels-1) + 2] += p2p_t;
                    setup_times[5*(num_levels-1) + 3] += vec_t;
                    setup_times[5*(num_levels-1) + 4] += mat_t;

                    solve_times = new double[5 * num_levels]();
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
                
            virtual void extend_hierarchy(int nrhs = 1) = 0;

            void duplicate_coarse()
            {
                int rank, num_procs;
                RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
                RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);

                int last_level = num_levels - 1;
                ParCSRMatrix* Ac = levels[last_level]->A;
                ParVector& b = levels[last_level]->b;
                aligned_vector<int> proc_sizes(num_procs);
                aligned_vector<int> active_procs;
                RAPtor_MPI_Allgather(&(Ac->local_num_rows), 1, RAPtor_MPI_INT, proc_sizes.data(),
                        1, RAPtor_MPI_INT, RAPtor_MPI_COMM_WORLD);
                for (int i = 0; i < num_procs; i++)
                {
                    if (proc_sizes[i])
                    {
                        active_procs.emplace_back(i);
                    }
                }
                RAPtor_MPI_Group world_group;
                RAPtor_MPI_Comm_group(RAPtor_MPI_COMM_WORLD, &world_group);                
                RAPtor_MPI_Group active_group;
                RAPtor_MPI_Group_incl(world_group, active_procs.size(), active_procs.data(),
                        &active_group);
                RAPtor_MPI_Comm_create_group(RAPtor_MPI_COMM_WORLD, active_group, 0, &coarse_comm);
                RAPtor_MPI_Group_free(&world_group);
                RAPtor_MPI_Group_free(&active_group);

                if (Ac->local_num_rows)
                {
                    int num_active, active_rank;
                    RAPtor_MPI_Comm_rank(coarse_comm, &active_rank);
                    RAPtor_MPI_Comm_size(coarse_comm, &num_active);

                    int proc;
                    int global_col, local_col;
                    int start, end;

                    aligned_vector<double> A_coarse_lcl;

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

                    aligned_vector<int> global_row_indices(coarse_displs[num_active]);

                    RAPtor_MPI_Allgatherv(Ac->local_row_map.data(), Ac->local_num_rows, RAPtor_MPI_INT,
                            global_row_indices.data(), coarse_sizes.data(), 
                            coarse_displs.data(), RAPtor_MPI_INT, coarse_comm);
    
                    std::map<int, int> global_to_local;
                    int ctr = 0;
                    for (aligned_vector<int>::iterator it = global_row_indices.begin();
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
                    
                    RAPtor_MPI_Allgatherv(A_coarse_lcl.data(), A_coarse_lcl.size(), RAPtor_MPI_DOUBLE,
                            A_coarse.data(), coarse_sizes.data(), coarse_displs.data(), 
                            RAPtor_MPI_DOUBLE, coarse_comm);

                    LU_permute.resize(coarse_n);
                    int info;
                    dgetrf_(&coarse_n, &coarse_n, A_coarse.data(), &coarse_n, 
                            LU_permute.data(), &info);

                    for (int i = 0; i < num_active; i++)
                    {
                        coarse_sizes[i] /= coarse_n;
                        coarse_displs[i+1] /= coarse_n;
                    }
                    
                    // Added for block vectors
                    for (int i = 0; i < num_active; i++)
                    {
                        coarse_sizes[i] *= b.local->b_vecs;
                        coarse_displs[i+1] *= b.local->b_vecs;
                    }
                }
            }

            // Stopped editing right here
            void cycle(ParVector& x, ParVector& b, int level = 0)
            {
                if (solve_times)
                {
                    init_profile();
                }

                ParCSRMatrix* A = levels[level]->A;
                ParCSRMatrix* P = levels[level]->P;
                ParVector& tmp = levels[level]->tmp;
                bool tap_level = tap_amg >= 0 && tap_amg <= level;

                if (level == num_levels - 1)
                {
                    if (A->local_num_rows)
                    {
                        int active_rank;
                        RAPtor_MPI_Comm_rank(coarse_comm, &active_rank);

                        char trans = 'N'; //No transpose
                        int nhrs = b.local->b_vecs; // Number of right hand sides
                        int info; // result

                        aligned_vector<double> b_data_temp(coarse_n*nhrs);
                        RAPtor_MPI_Allgatherv(b.local->data(), b.local_n*nhrs, RAPtor_MPI_DOUBLE,
                                b_data_temp.data(), coarse_sizes.data(), coarse_displs.data(), 
                                RAPtor_MPI_DOUBLE, coarse_comm);

                        // Reorder b_data_temp into correct rhs for block coarse solve
                        aligned_vector<double> b_data;
                        int start, stop;
                        for (int v = 0; v < nhrs; v++)
                        {
                            for (int i = 0; i < coarse_sizes.size(); i++)
                            {
                                start = coarse_displs[i] + (v * coarse_sizes[i] / nhrs);
                                stop = start + (coarse_sizes[i] / nhrs);
                                for (int j = start; j < stop; j++)
                                {
                                    b_data.emplace_back(b_data_temp[j]);
                                }
                            }
                        }

                        dgetrs_(&trans, &coarse_n, &nhrs, A_coarse.data(), &coarse_n, 
                                LU_permute.data(), b_data.data(), &coarse_n, &info);

                        for (int v = 0; v < nhrs; v++)
                        {
                            for (int i = 0; i < b.local_n; i++)
                            {
                                x.local->values[i + v*b.local_n] = 
                                    b_data[i + (coarse_displs[active_rank]/nhrs) + v*b.global_n];
                            }
                        }
                    }

                    if (solve_times)
                    {
                        finalize_profile();
                        solve_times[5*level] += total_t;
                        solve_times[5*level + 1] += collective_t;
                        solve_times[5*level + 2] += p2p_t;
                        solve_times[5*level + 3] += vec_t;
                        solve_times[5*level + 4] += mat_t;
                    }
                }
                else
                {
                    levels[level+1]->x.set_const_value(0.0);
                    
                    // Relax
                    switch (relax_type)
                    {
                        case Jacobi:
                            jacobi(A, x, b, tmp, num_smooth_sweeps, relax_weight,
                                    tap_level);
                            break;
                        case SOR:
                            sor(A, x, b, tmp, num_smooth_sweeps, relax_weight,
                                    tap_level);
                            break;
                        case SSOR:
                            ssor(A, x, b, tmp, num_smooth_sweeps, relax_weight,
                                    tap_level);
                            break;
                    }


                    A->residual(x, b, tmp, tap_level);

                    P->mult_T(tmp, levels[level+1]->b, tap_level);


                    if (solve_times)
                    {
                        finalize_profile();
                        solve_times[5*level] += total_t;
                        solve_times[5*level + 1] += collective_t;
                        solve_times[5*level + 2] += p2p_t;
                        solve_times[5*level + 3] += vec_t;
                        solve_times[5*level + 4] += mat_t;
                    }
                    cycle(levels[level+1]->x, levels[level+1]->b, level+1);
                    if (solve_times)
                    {
                        init_profile();
                    }

                    P->mult_append(levels[level+1]->x, x, tap_level);
                        
                    /*for (int p = 0; p < num_procs; p++)
                    {
                        if (p == rank)
                        {
                            printf("%d mult_append x\n", rank);
                            for (int v = 0; v < x.local->b_vecs; v++)
                            {
                                printf("v %d ", v);
                                for (int i = 0; i < x.local_n; i++)
                                {
                                    printf("%e ", x.local->values[v*x.local_n + i]);
                                }
                                printf("\n");
                            }
                        }
                        MPI_Barrier(MPI_COMM_WORLD);
                    }*/

                    switch (relax_type)
                    {
                        case Jacobi:
                            jacobi(A, x, b, tmp, num_smooth_sweeps, relax_weight,
                                    tap_level);
                            break;
                        case SOR:
                            sor(A, x, b, tmp, num_smooth_sweeps, relax_weight,
                                    tap_level);
                            break;
                        case SSOR:
                            ssor(A, x, b, tmp, num_smooth_sweeps, relax_weight,
                                    tap_level);
                            break;
                     }
                    if (solve_times)
                    {
                        finalize_profile();
                        solve_times[5*level] += total_t;
                        solve_times[5*level + 1] += collective_t;
                        solve_times[5*level + 2] += p2p_t;
                        solve_times[5*level + 3] += vec_t;
                        solve_times[5*level + 4] += mat_t;
                    }
                }
            }

            int solve(ParVector& sol, ParVector& rhs)
            {
                double b_norm;
                aligned_vector<double> b_norms;
                if (rhs.local->b_vecs > 1)
                {
                    b_norms.resize(rhs.local->b_vecs);
                    b_norm = rhs.norm(2, &(b_norms[0]));
                }
                else b_norm = rhs.norm(2);

                double r_norm;
                aligned_vector<double> r_norms;
                int iter = 0;

                if (store_residuals)
                {
                    residuals.resize(max_iterations + 1 * rhs.local->b_vecs);
                }

                if (track_times)
                {
                    if (!solve_times) solve_times = new double[5*num_levels]();
                    init_profile();
                }

                // Iterate until convergence or max iterations
                //ParVector resid(rhs.global_n, rhs.local_n);
                ParBVector resid(rhs.global_n, rhs.local_n, rhs.local->b_vecs);
                levels[0]->A->residual(sol, rhs, resid);
                if (rhs.local->b_vecs > 1)
                {
                    r_norms.resize(rhs.local->b_vecs);
                    r_norm = resid.norm(2, &(r_norms[0]));
                    for (int i = 0; i < rhs.local->b_vecs; i++)
                    {
                        if (fabs(b_norms[i]) > zero_tol) r_norms[i] = r_norms[i] / b_norms[i];
                        if (r_norms[i] > r_norm) r_norm = r_norms[i];
                    }
                    if (store_residuals)
                    {
                        int start_indx = iter * rhs.local->b_vecs;
                        for (int i = 0; i < rhs.local->b_vecs; i++)
                        {
                            residuals[start_indx + i] = r_norms[i];
                        }
                    }
                }
                else
                {
                    r_norm = resid.norm(2);
                    if (fabs(b_norm) > zero_tol) r_norm = r_norm / b_norm;
                    if (store_residuals) residuals[iter] = r_norm;
                }

                if (track_times)
                {
                    finalize_profile();
                    solve_times[0] += total_t;
                    solve_times[1] += collective_t;
                    solve_times[2] += p2p_t;
                    solve_times[3] += vec_t;
                    solve_times[4] += mat_t;
                }

                while (r_norm > solve_tol && iter < max_iterations)
                {
                    // sol and rhs correct before cycle called
                    cycle(sol, rhs, 0);

                    if (track_times)
                    {
                        init_profile();
                    }

                    iter++;
                    levels[0]->A->residual(sol, rhs, resid);
                    if (rhs.local->b_vecs > 1)
                    {
                        r_norm = resid.norm(2, &(r_norms[0]));
                        for (int i = 0; i < rhs.local->b_vecs; i++)
                        {
                            if (fabs(b_norms[i]) > zero_tol) r_norms[i] = r_norms[i] / b_norms[i];
                            if (r_norms[i] > r_norm) r_norm = r_norms[i];
                        }
                        if (store_residuals)
                        {
                            int start_indx = iter * rhs.local->b_vecs;
                            for (int i = 0; i < rhs.local->b_vecs; i++)
                            {
                                residuals[start_indx + i] = r_norms[i];
                            }
                        }
                        //printf("residuals %e %e %e\n", r_norms[0], r_norms[1], r_norms[2]);
                    }
                    else
                    {
                        r_norm = resid.norm(2);
                        if (fabs(b_norm) > zero_tol) r_norm = r_norm / b_norm;
                        if (store_residuals) residuals[iter] = r_norm;
                        //printf("residual %e\n", r_norm);
                    }

                    if (track_times)
                    {
                        finalize_profile();
                        solve_times[0] += total_t;
                        solve_times[1] += collective_t;
                        solve_times[2] += p2p_t;
                        solve_times[3] += vec_t;
                        solve_times[4] += mat_t;
                    }
                }


                return iter;
            }

            void print_hierarchy()
            {
                int rank;
                RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);

                if (rank == 0)
                {
                    printf("Num Levels = %d\n", num_levels);
                    printf("A\tNRow\tNCol\tNNZ\n");
                }

                for (int i = 0; i < num_levels; i++)
                {
                    ParCSRMatrix* Al = levels[i]->A;
                    long lcl_nnz = Al->local_nnz;
                    long nnz;
                    RAPtor_MPI_Reduce(&lcl_nnz, &nnz, 1, RAPtor_MPI_LONG, RAPtor_MPI_SUM, 0, RAPtor_MPI_COMM_WORLD);
                    if (rank == 0)
                    {
                        printf("%d\t%d\t%d\t%lu\n", i, 
                                Al->global_num_rows, Al->global_num_cols, nnz);
                    }
                }
            }

            void print_residuals(int iter)
            {
                int rank;
                RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
                if (rank == 0) 
                {
                    for (int i = 0; i < iter + 1; i++)
                    {
                        printf("Res[%d] = %e\n", i, residuals[i]);
                    }
                }
            }

            void print_times(double* times, const char* phase)
            {
                if (times == NULL) return;

                int rank;
                RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);

                double max_t;
                for (int i = 0; i < num_levels; i++)
                {
                    if (rank == 0) printf("Level %d\n", i);

                    RAPtor_MPI_Reduce(&times[5*i], &max_t, 1, RAPtor_MPI_DOUBLE, 
                            RAPtor_MPI_MAX, 0, RAPtor_MPI_COMM_WORLD);
                    if (rank == 0 && max_t > 0) printf("%s Total Time: %e\n", phase, max_t);

                    RAPtor_MPI_Reduce(&times[5*i+1], &max_t, 1, RAPtor_MPI_DOUBLE, 
                            RAPtor_MPI_MAX, 0, RAPtor_MPI_COMM_WORLD);
                    if (rank == 0 && max_t > 0) printf("%s Collective Time: %e\n", phase, max_t);

                    RAPtor_MPI_Reduce(&times[5*i+2], &max_t, 1, RAPtor_MPI_DOUBLE, 
                            RAPtor_MPI_MAX, 0, RAPtor_MPI_COMM_WORLD);
                    if (rank == 0 && max_t > 0) printf("%s P2P Time: %e\n", phase, max_t);

                    RAPtor_MPI_Reduce(&times[5*i+3], &max_t, 1, RAPtor_MPI_DOUBLE, 
                            RAPtor_MPI_MAX, 0, RAPtor_MPI_COMM_WORLD);
                    if (rank == 0 && max_t > 0) printf("%s Vec Comm Time: %e\n", phase, max_t);

                    RAPtor_MPI_Reduce(&times[5*i+4], &max_t, 1, RAPtor_MPI_DOUBLE, 
                            RAPtor_MPI_MAX, 0, RAPtor_MPI_COMM_WORLD);
                    if (rank == 0 && max_t > 0) printf("%s Mat Comm Time: %e\n", phase, max_t);
                }
            }

            void print_setup_times()
            {
                print_times(setup_times, "Setup");
            }
            void print_solve_times()
            {
                print_times(solve_times, "Solve");
            }

            aligned_vector<double>& get_residuals()
            {
                return residuals;
            }

            strength_t strength_type;
            relax_t relax_type;

            int num_smooth_sweeps;
            int max_coarse;
            int max_levels;
            int tap_amg;
            int tap_simple;
            int max_iterations;

            double strong_threshold;
            double relax_weight;
            double sparsify_tol;
            double solve_tol;

            bool store_residuals;

            double* weights;
            aligned_vector<double> residuals;

            std::vector<ParLevel*> levels;
            aligned_vector<int> LU_permute;
            int num_levels;
            int num_variables;
            
            bool track_times;
            double* setup_times;
            double* solve_times;

            int coarse_n;
            aligned_vector<double> A_coarse;
            aligned_vector<int> coarse_sizes;
            aligned_vector<int> coarse_displs;
            RAPtor_MPI_Comm coarse_comm;
    };
}
#endif
