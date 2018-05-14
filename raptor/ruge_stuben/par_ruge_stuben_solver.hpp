// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_PAR_RUGE_STUBEN_SOLVER_HPP
#define RAPTOR_PAR_RUGE_STUBEN_SOLVER_HPP

#include "multilevel/par_multilevel.hpp"
#include "ruge_stuben/par_cf_splitting.hpp"
#include "ruge_stuben/par_interpolation.hpp"

namespace raptor
{
    class ParRugeStubenSolver : public ParMultilevel
    {
      public:
        ParRugeStubenSolver(double _strong_threshold = 0.0, coarsen_t _coarsen_type = RS, 
                interp_t _interp_type = Direct, strength_t _strength_type = Classical,
                relax_t _relax_type = SOR) 
            : ParMultilevel(_strong_threshold, _strength_type, _relax_type)
        {
            coarsen_type = _coarsen_type;
            interp_type = _interp_type;
            variables = NULL;
            num_variables = 1;
        }

        ~ParRugeStubenSolver()
        {

        }

        void setup(ParCSRMatrix *Af)
        {
            if (num_variables > 1 && variables == NULL) 
            {
                form_variable_list(Af, num_variables);
            }

            setup_helper(Af);

            delete[] variables;
            variables = NULL;
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

       void extend_hierarchy()
        {
            int level_ctr = levels.size() - 1;
            bool tap_level = tap_amg >= 0 && tap_amg <= level_ctr;

            ParCSRMatrix* A = levels[level_ctr]->A;
            ParCSRMatrix* S;
            ParCSRMatrix* P;
            ParCSRMatrix* AP;
            ParCSCMatrix* P_csc;

            aligned_vector<int> states;
            aligned_vector<int> off_proc_states;

            // Form strength of connection
            strength_times[level_ctr] -= MPI_Wtime();
            S = A->strength(strength_type, strong_threshold, num_variables, variables);
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
                case ModClassical:
                    P = mod_classical_interpolation(A, S, states, off_proc_states, 
                            false, num_variables, variables);
                    break;
                case Extended:
                    P = extended_interpolation(A, S, states, off_proc_states, 
                            tap_level, num_variables, variables);
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

        coarsen_t coarsen_type;
        interp_t interp_type;

        int* variables;

    };
}
   

#endif


