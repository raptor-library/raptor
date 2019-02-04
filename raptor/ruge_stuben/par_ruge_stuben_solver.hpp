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

            if (num_variables > 1) delete[] variables;
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

            aligned_vector<int> states;
            aligned_vector<int> off_proc_states;

            // Form strength of connection
            S = A->strength(strength_type, strong_threshold, tap_level, 
                    num_variables, variables);

            // Form CF Splitting
            switch (coarsen_type)
            {
                case RS:
                    if (level_ctr < 3) 
                    {
                        split_rs(S, states, off_proc_states, tap_level);
                    }
                    else 
                    {
                        split_falgout(S, states, off_proc_states, tap_level, 
                                weights);
                    }
                    break;
                case CLJP:
                    split_cljp(S, states, off_proc_states, tap_level, 
                            weights);
                    break;
                case Falgout:
                    split_falgout(S, states, off_proc_states, tap_level, 
                            weights);
                    break;
                case PMIS:
                    split_pmis(S, states, off_proc_states, tap_level, 
                            weights);
                    break;
                case HMIS:
                    split_hmis(S, states, off_proc_states, tap_level, 
                            weights);
                    break;
            }

            // Form modified classical interpolation
            switch (interp_type)
            {
                case Direct:
                    P = direct_interpolation(A, S, states, off_proc_states, 
                            tap_level);
                    break;
                case ModClassical:
                    P = mod_classical_interpolation(A, S, states, off_proc_states, 
                            tap_level, num_variables, variables);
                    break;
                case Extended:
                    P = extended_interpolation(A, S, states, off_proc_states, 
                            tap_level, num_variables, variables);
                    break;
            }
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
            levels.emplace_back(new ParLevel());

            AP = A->mult(levels[level_ctr]->P, tap_level);
            A = AP->mult_T(P, tap_level);

            A->sort();
            A->on_proc->move_diag();

            level_ctr++;
            levels[level_ctr]->A = A;
            A->comm = new ParComm(A->partition, A->off_proc_column_map,
                    A->on_proc_column_map, levels[level_ctr-1]->A->comm->key,
                    levels[level_ctr-1]->A->comm->mpi_comm);
            levels[level_ctr]->x.resize(A->global_num_rows, A->local_num_rows);
            levels[level_ctr]->b.resize(A->global_num_rows, A->local_num_rows);
            levels[level_ctr]->tmp.resize(A->global_num_rows, A->local_num_rows);
            levels[level_ctr]->P = NULL;

            if (tap_amg >= 0 && tap_amg <= level_ctr)
            {
                levels[level_ctr]->A->init_tap_communicators(RAPtor_MPI_COMM_WORLD);
            }

            delete AP;
            delete S;
        }    

        coarsen_t coarsen_type;
        interp_t interp_type;

        int* variables;

    };
}
   

#endif


