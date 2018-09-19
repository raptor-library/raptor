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
            if (track_times)
            {
                n_setup_times = 6;
                setup_times = new aligned_vector<double>[n_setup_times];
                setup_comm_times = new aligned_vector<double>[n_setup_times];
                setup_mat_comm_times = new aligned_vector<double>[n_setup_times];
            }

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

            double* total_time = NULL;
            double* strength_time = NULL;
            double* coarsen_time = NULL;
            double* interp_time = NULL;
            double* AP_time = NULL;
            double* PTAP_time = NULL;

            double* total_mat_time = NULL;
            double* strength_mat_time = NULL;
            double* coarsen_mat_time = NULL;
            double* interp_mat_time = NULL;
            double* AP_mat_time = NULL;
            double* PTAP_mat_time = NULL;
            if (setup_times) 
            {
                setup_times[0][level_ctr] -= MPI_Wtime();
                total_time = &setup_comm_times[0][level_ctr];
                strength_time = &setup_comm_times[1][level_ctr];
                coarsen_time = &setup_comm_times[2][level_ctr];
                interp_time = &setup_comm_times[3][level_ctr];
                AP_time = &setup_comm_times[4][level_ctr];
                PTAP_time = &setup_comm_times[5][level_ctr];

                total_mat_time = &setup_mat_comm_times[0][level_ctr];
                strength_mat_time = &setup_mat_comm_times[1][level_ctr];
                coarsen_mat_time = &setup_mat_comm_times[2][level_ctr];
                interp_mat_time = &setup_mat_comm_times[3][level_ctr];
                AP_mat_time = &setup_mat_comm_times[4][level_ctr];
                PTAP_mat_time = &setup_mat_comm_times[5][level_ctr];
            }

            ParCSRMatrix* A = levels[level_ctr]->A;
            ParCSRMatrix* S;
            ParCSRMatrix* P;
            ParCSRMatrix* AP;

            aligned_vector<int> states;
            aligned_vector<int> off_proc_states;

            // Form strength of connection
            if (setup_times) setup_times[1][level_ctr] -= MPI_Wtime();
            S = A->strength(strength_type, strong_threshold, tap_level, 
                    num_variables, variables, strength_time);
            if (setup_times) setup_times[1][level_ctr] += MPI_Wtime();

            // Form CF Splitting
            if (setup_times) setup_times[2][level_ctr] -= MPI_Wtime();

            aligned_vector<int> A_to_S(A->off_proc_num_cols, -1);
            int ctr = 0;
            for (int i = 0; i < S->off_proc_num_cols; i++)
            {
                int global_col = S->off_proc_column_map[i];
                while (A->off_proc_column_map[ctr] != global_col)
                ctr++;
                A_to_S[ctr] = i;
            }
            if (tap_level) S->tap_comm = new TAPComm((TAPComm*) A->tap_comm, A_to_S, coarsen_time);
            S->comm = new ParComm((ParComm*) A->comm, A_to_S, coarsen_time);

            switch (coarsen_type)
            {
                case RS:
                    if (level_ctr < 3) 
                    {
                        split_rs(S, states, off_proc_states, tap_level, coarsen_time);
                    }
                    else 
                    {
                        split_falgout(S, states, off_proc_states, tap_level, 
                                weights, coarsen_time, coarsen_mat_time);
                    }
                    break;
                case CLJP:
                    split_cljp(S, states, off_proc_states, tap_level, 
                            weights, coarsen_time, coarsen_mat_time);
                    break;
                case Falgout:
                    split_falgout(S, states, off_proc_states, tap_level, 
                            weights, coarsen_time, coarsen_mat_time);
                    break;
                case PMIS:
                    split_pmis(S, states, off_proc_states, tap_level, 
                            weights, coarsen_time);
                    break;
                case HMIS:
                    split_hmis(S, states, off_proc_states, tap_level, 
                            weights, coarsen_time);
                    break;
            }
            if (setup_times) setup_times[2][level_ctr] += MPI_Wtime();

            // Form modified classical interpolation
            if (setup_times) setup_times[3][level_ctr] -= MPI_Wtime();
            switch (interp_type)
            {
                case Direct:
                    P = direct_interpolation(A, S, states, off_proc_states, 
                            tap_level, interp_time);
                    break;
                case ModClassical:
                    P = mod_classical_interpolation(A, S, states, off_proc_states, 
                            tap_level, num_variables, variables, interp_time,
                            interp_mat_time);
                    break;
                case Extended:
                    P = extended_interpolation(A, S, states, off_proc_states, 
                            tap_level, num_variables, variables, interp_time,
                            interp_mat_time);
                    break;
            }
            if (setup_times) setup_times[3][level_ctr] += MPI_Wtime();
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


            if (setup_times) setup_times[4][level_ctr] -= MPI_Wtime();
            AP = A->mult(levels[level_ctr]->P, tap_level, AP_mat_time);
            if (setup_times) setup_times[4][level_ctr] += MPI_Wtime();

            if (setup_times) setup_times[5][level_ctr] -= MPI_Wtime();
            A = AP->mult_T(P, tap_level, PTAP_mat_time);
            if (setup_times) setup_times[5][level_ctr] += MPI_Wtime();

            level_ctr++;
            levels[level_ctr]->A = A;
            A->comm = new ParComm(A->partition, A->off_proc_column_map,
                    A->on_proc_column_map, levels[level_ctr-1]->A->comm->key,
                    levels[level_ctr-1]->A->comm->mpi_comm, total_time);
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
                        levels[level_ctr]->A->on_proc_column_map,
                        true, MPI_COMM_WORLD, total_time);
            }

            delete AP;
            delete S;

            if (setup_times) 
            {
                setup_times[0][level_ctr-1] += MPI_Wtime();
                *total_time += (*strength_time + *coarsen_time + *interp_time + 
                        *AP_time + *PTAP_time);
                *total_mat_time += (*strength_mat_time + *coarsen_mat_time + 
                        *interp_mat_time + *AP_mat_time + *PTAP_mat_time);
            }
        }    

        void print_setup_times()
        {
            if (setup_times == NULL) return;

            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            double max_t;
            for (int i = 0; i < num_levels; i++)
            {
                if (rank == 0) printf("Level %d\n", i);

                MPI_Reduce(&setup_times[0][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Setup Time: %e\n", max_t);

                MPI_Reduce(&setup_comm_times[0][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Setup Vec Comm Time: %e\n", max_t);

                MPI_Reduce(&setup_mat_comm_times[0][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Setup Mat Comm Time: %e\n", max_t);

                MPI_Reduce(&setup_times[1][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Strength: %e\n", max_t);

                MPI_Reduce(&setup_comm_times[1][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Strength Vec Comm: %e\n", max_t);

                MPI_Reduce(&setup_mat_comm_times[1][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Strength Mat Comm: %e\n", max_t);

                MPI_Reduce(&setup_times[2][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("C/F Splitting: %e\n", max_t);

                MPI_Reduce(&setup_comm_times[2][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("C/F Splitting Vec Comm: %e\n", max_t);

                MPI_Reduce(&setup_mat_comm_times[2][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("C/F Splitting Mat Comm: %e\n", max_t);

                MPI_Reduce(&setup_times[3][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Form Interp: %e\n", max_t);

                MPI_Reduce(&setup_comm_times[3][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Form Interp Vec Comm: %e\n", max_t);

                MPI_Reduce(&setup_mat_comm_times[3][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Form Interp Mat Comm: %e\n", max_t);

                MPI_Reduce(&setup_times[4][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("A*P: %e\n", max_t);

                MPI_Reduce(&setup_comm_times[4][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("A*P Vec Comm: %e\n", max_t);

                MPI_Reduce(&setup_mat_comm_times[4][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("A*P Mat Comm: %e\n", max_t);

                MPI_Reduce(&setup_times[5][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("P.T*AP: %e\n", max_t);

                MPI_Reduce(&setup_comm_times[5][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("P.T*AP Vec Comm: %e\n", max_t);

                MPI_Reduce(&setup_mat_comm_times[5][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("P.T*AP Mat Comm: %e\n", max_t);

            }
        }

        coarsen_t coarsen_type;
        interp_t interp_type;

        int* variables;

    };
}
   

#endif


