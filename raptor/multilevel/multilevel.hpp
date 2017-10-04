// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_MULTILEVEL_H
#define RAPTOR_CORE_MULTILEVEL_H

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "multilevel/level.hpp"
#include "util/linalg/relax.hpp"
#include "ruge_stuben/par_interpolation.hpp"
#include "ruge_stuben/par_cf_splitting.hpp"
//#include "aggregation/prolongation.hpp"

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

            Multilevel(ParCSRMatrix* Af, double theta = 0.0, int num_smooth_steps = 1, 
                    int max_coarse = 50, int max_levels = -1)
            {
                int rank, num_procs;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

                int last_level = 0;

                // Add original, fine level to hierarchy
                levels.push_back(new Level());
                levels[0]->A = new ParCSRMatrix(Af);
                levels[0]->A->sort();
                levels[0]->A->on_proc->move_diag();
                levels[0]->x.resize(Af->global_num_rows, Af->local_num_rows,
                        Af->partition->first_local_row);
                levels[0]->b.resize(Af->global_num_rows, Af->local_num_rows,
                        Af->partition->first_local_row);
                levels[0]->tmp.resize(Af->global_num_rows, Af->local_num_rows,
                        Af->partition->first_local_row);

                // Add coarse levels to hierarchy 
                while (levels[last_level]->A->global_num_rows > max_coarse && 
                        (max_levels == -1 || levels.size() < max_levels))
                {
                    extend_hierarchy(theta, num_smooth_steps);
                    last_level++;
                }
                num_levels = levels.size();

                // Duplicate coarsest level across all processes that hold any
                // rows of A_c
                duplicate_coarse();
            }

            // Creates Smoothed Aggregation Solver -- TODO
            /*Multilevel(ParCSRMatrix* Af, data_t* B_ptr = NULL, int num_candidates = 1,
                    double theta = 0.0, double omega = 4.0/3, 
                    int num_smooth_steps = 1, int max_coarse = 50)
            {
                // Always need levels with A, x, b (P and tmp are on all but
                // coarsest)
                levels.push_back(new Level());
                levels[0]->A = new ParCSRMatrix(Af);
                levels[0]->A->sort();
                levels[0]->x.resize(Af->global_num_rows, Af->local_num_rows,
                        Af->first_local_row);
                levels[0]->b.resize(Af->global_num_rows, Af->local_num_rows,
                        Af->first_local_row);
                levels[0]->tmp.resize(Af->global_num_rows, Af->local_num_rows,
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
            }*/



            ~Multilevel()
            {
                if (levels[num_levels-1]->A->local_num_rows)
                {
                    MPI_Comm_free(&coarse_comm);
                }
                for (std::vector<Level*>::iterator it = levels.begin();
                        it != levels.end(); ++it)
                {
                    delete *it;
                }
            }

            void extend_hierarchy(double theta = 0.0, int num_smooth_steps = 1)
            {
                int level_ctr = levels.size() - 1;
                ParCSRMatrix* A = levels[level_ctr]->A;
                ParCSRMatrix* S;
                ParCSRMatrix* AP;
                ParCSCMatrix* P_csc;

                std::vector<int> states;
                std::vector<int> off_proc_states;

                S = A->strength(theta);
                split_falgout(S, states, off_proc_states);
                levels[level_ctr]->P = direct_interpolation(A, S, states, off_proc_states);
                ParCSRMatrix* P = levels[level_ctr]->P;

                levels.push_back(new Level());
                AP = A->mult(levels[level_ctr]->P);
                P_csc = new ParCSCMatrix(levels[level_ctr]->P);
                
                level_ctr++;
                levels[level_ctr]->A = AP->mult_T(P_csc);
                A = levels[level_ctr]->A;
                A->comm = new ParComm(A->partition, A->off_proc_column_map,
                        A->on_proc_column_map);
                levels[level_ctr]->x.resize(A->global_num_rows, A->local_num_rows,
                        A->partition->first_local_row);
                levels[level_ctr]->b.resize(A->global_num_rows, A->local_num_rows,
                        A->partition->first_local_row);
                levels[level_ctr]->tmp.resize(A->global_num_rows, A->local_num_rows,
                        A->partition->first_local_row);
                levels[level_ctr]->P = NULL;

                delete AP;
                delete P_csc;
                delete S;
            }

/*            data_t* extend_hierarchy(data_t* B,
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
                levels[level_ctr]->x.resize(levels[level_ctr]->A->n_rows);
                levels[level_ctr]->b.resize(levels[level_ctr]->A->n_rows);
                levels[level_ctr]->tmp.resize(levels[level_ctr]->A->n_rows);
                
                return R;
            }*/


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

            void cycle(int level)
            {
                ParCSRMatrix* A = levels[level]->A;
                ParCSRMatrix* P = levels[level]->P;
                ParVector& x = levels[level]->x;
                ParVector& b = levels[level]->b;
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
                else
                {
                    levels[level+1]->x.set_const_value(0.0);
                    relax(levels[level]);
                    A->residual(x, b, tmp);
                    P->mult_T(tmp, levels[level+1]->b);
                    cycle(level+1);
                    P->mult(levels[level+1]->x, tmp);
                    for (int i = 0; i < A->local_num_rows; i++)
                    {
                        x.local[i] += tmp.local[i];
                    }
                    relax(levels[level]);
                }
            }

            void solve(ParVector& sol, ParVector& rhs, int num_iterations = 100)
            {
                int rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);

                double b_norm = rhs.norm(2);
                double r_norm;
                int iter = 0;

                levels[0]->x.copy(sol);
                levels[0]->b.copy(rhs);

                // Iterate until convergence or max iterations
                ParVector resid(rhs.global_n, rhs.local_n, rhs.first_local);
                levels[0]->A->residual(levels[0]->x, levels[0]->b, resid);
                if (fabs(b_norm) > zero_tol)
                {
                    r_norm = resid.norm(2) / b_norm;
                }
                else
                {
                    r_norm = resid.norm(2);
                    if (rank == 0) printf("Small Bnorm -> not using relative residual\n");
                }
                if (rank == 0) printf("Rnorm = %e\n", r_norm);

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
                    if (rank == 0) printf("Rnorm = %e\n", r_norm);
                }
            } 

            std::vector<Level*> levels;
            std::vector<int> LU_permute;
            int num_levels;

            int coarse_n;
            std::vector<double> A_coarse;
            std::vector<int> coarse_sizes;
            std::vector<int> coarse_displs;
            MPI_Comm coarse_comm;
    };
}
#endif
