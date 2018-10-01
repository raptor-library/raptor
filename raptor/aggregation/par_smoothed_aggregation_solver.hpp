// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_PAR_SMOOTHED_AGGREGATION_SOLVER_HPP
#define RAPTOR_PAR_SMOOTHED_AGGREGATION_SOLVER_HPP

#include "multilevel/par_multilevel.hpp"
#include "aggregation/par_mis.hpp"
#include "aggregation/par_aggregate.hpp"
#include "aggregation/par_candidates.hpp"
#include "aggregation/par_prolongation.hpp"

namespace raptor
{
    class ParSmoothedAggregationSolver : public ParMultilevel
    {
      public:
        ParSmoothedAggregationSolver(double _strong_threshold = 0.0, agg_t _agg_type = MIS, 
                prolong_t _prolong_type = JacobiProlongation,
                strength_t _strength_type = Symmetric,
                relax_t _relax_type = SOR,
                int _prolong_smooth_steps = 1, 
                double _prolong_weight = 4.0/3) 
            : ParMultilevel(_strong_threshold, _strength_type, _relax_type)
        {
            agg_type = _agg_type;
            prolong_type = _prolong_type;
            num_candidates = 1;
            interp_tol = 1e-10;
            prolong_smooth_steps = _prolong_smooth_steps;
            prolong_weight = _prolong_weight;
        }

        ~ParSmoothedAggregationSolver()
        {
        }

        void setup(ParCSRMatrix* Af) 
        {
            if (track_times)
            {
                n_setup_times = 7;
                setup_times = new aligned_vector<double>[n_setup_times];
                setup_comm_times = new aligned_vector<double>[n_setup_times];
                setup_mat_comm_times = new aligned_vector<double>[n_setup_times];
            }

            // TODO -- add option for B to be passed as variable
            num_candidates = 1;
            B.resize(Af->local_num_rows);
            for (int i = 0; i < Af->local_num_rows; i++)
            {
                B[i] = 1.0;
            }

            setup_helper(Af);
        }

        void extend_hierarchy()
        {
            int level_ctr = levels.size() - 1;
            bool tap_level = tap_amg >= 0 && tap_amg <= level_ctr;

            data_t* total_time = NULL;
            data_t* strength_time = NULL;
            data_t* agg_time = NULL;
            data_t* interp_time = NULL;
            data_t* prolong_time = NULL;
            data_t* AP_time = NULL;
            data_t* PTAP_time = NULL;

            data_t* total_mat_time = NULL;
            data_t* strength_mat_time = NULL;
            data_t* agg_mat_time = NULL;
            data_t* interp_mat_time = NULL;
            data_t* prolong_mat_time = NULL;
            data_t* AP_mat_time = NULL;
            data_t* PTAP_mat_time = NULL;
            if (setup_times)
            {
                setup_times[0][level_ctr] -= MPI_Wtime();
                total_time = &setup_comm_times[0][level_ctr];
                strength_time = &setup_comm_times[1][level_ctr];
                agg_time = &setup_comm_times[2][level_ctr];
                interp_time = &setup_comm_times[3][level_ctr];
                prolong_time = &setup_comm_times[4][level_ctr];
                AP_time = &setup_comm_times[5][level_ctr];
                PTAP_time = &setup_comm_times[6][level_ctr];

                total_mat_time = &setup_mat_comm_times[0][level_ctr];
                strength_mat_time = &setup_mat_comm_times[1][level_ctr];
                agg_mat_time = &setup_mat_comm_times[2][level_ctr];
                interp_mat_time = &setup_mat_comm_times[3][level_ctr];
                prolong_mat_time = &setup_mat_comm_times[4][level_ctr];
                AP_mat_time = &setup_mat_comm_times[5][level_ctr];
                PTAP_mat_time = &setup_mat_comm_times[6][level_ctr];
            }

            ParCSRMatrix* A = levels[level_ctr]->A;
            ParCSRMatrix* S;
            ParCSRMatrix* T;
            ParCSRMatrix* P;
            ParCSRMatrix* AP;

            aligned_vector<int> states;
            aligned_vector<int> off_proc_states;
            aligned_vector<int> aggregates;
            aligned_vector<double> R;
            int n_aggs;

            // Form strength of connection
            if (setup_times) setup_times[1][level_ctr] -= MPI_Wtime();
            S = A->strength(strength_type, strong_threshold, tap_level, 
                    1, NULL, strength_time);
            if (setup_times) setup_times[1][level_ctr] += MPI_Wtime();

            // Aggregate Nodes
            if (setup_times) setup_times[2][level_ctr] -= MPI_Wtime();
            switch (agg_type)
            {
                case MIS:
                    aligned_vector<int> A_to_S(A->off_proc_num_cols, -1);
                    int ctr = 0;
                    for (int i = 0; i < S->off_proc_num_cols; i++)
                    {
                        int global_col = S->off_proc_column_map[i];
                        while (A->off_proc_column_map[ctr] != global_col)
                        ctr++;
                        A_to_S[ctr] = i;
                    }
                    if (tap_level) S->tap_comm = new TAPComm((TAPComm*) A->tap_comm, A_to_S, agg_time);
                    S->comm = new ParComm((ParComm*) A->comm, A_to_S, agg_time);

                    mis2(S, states, off_proc_states, tap_level, weights, agg_time);
                    n_aggs = aggregate(A, S, states, off_proc_states, 
                            aggregates, tap_level, NULL, agg_time);
                    break;
            }
            if (setup_times) setup_times[2][level_ctr] += MPI_Wtime();

            // Form modified classical interpolation
            if (setup_times) setup_times[3][level_ctr] -= MPI_Wtime();
            // Form tentative interpolation
            T = fit_candidates(A, n_aggs, aggregates, B, R, 
                    num_candidates, false, interp_tol, interp_time);
            if (setup_times) setup_times[3][level_ctr] += MPI_Wtime();
            

            if (setup_times) setup_times[4][level_ctr] -= MPI_Wtime();
            switch (prolong_type)
            {
                case JacobiProlongation:
                    P = jacobi_prolongation(A, T, tap_level, 
                            prolong_weight, prolong_smooth_steps, prolong_time,
                            prolong_mat_time);
                    break;
            }
            if (setup_times) setup_times[4][level_ctr] += MPI_Wtime();
            levels[level_ctr]->P = P;

            // Form coarse grid operator
            levels.emplace_back(new ParLevel());

            if (setup_times) setup_times[5][level_ctr] -= MPI_Wtime();
            AP = A->mult(levels[level_ctr]->P, tap_level, AP_mat_time);
            if (setup_times) setup_times[5][level_ctr] += MPI_Wtime();

            if (setup_times) setup_times[6][level_ctr] -= MPI_Wtime();
            A = AP->mult_T(P, tap_level, PTAP_mat_time);
            if (setup_times) setup_times[6][level_ctr] += MPI_Wtime();

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
                // Create 2-step node-aware communicator for setup phase
                // will be changed to 3-step before solve phase
                levels[level_ctr]->A->tap_comm = new TAPComm(
                        levels[level_ctr]->A->partition,
                        levels[level_ctr]->A->off_proc_column_map,
                        levels[level_ctr]->A->on_proc_column_map, 
                        true, A->comm->mpi_comm, total_time);
            }

            std::copy(R.begin(), R.end(), B.begin());

            delete AP;
            delete T;
            delete S;

            if (setup_times) 
            {
                setup_times[0][level_ctr-1] += MPI_Wtime();
                *total_time += (*strength_time + *agg_time + *interp_time + 
                        *prolong_time + *AP_time + *PTAP_time);
                *total_mat_time += (*strength_mat_time + *agg_mat_time + 
                        *interp_mat_time + *prolong_mat_time + 
                        *AP_mat_time + *PTAP_mat_time);
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
                if (rank == 0 && max_t > 0) printf("Aggregate: %e\n", max_t);

                MPI_Reduce(&setup_comm_times[2][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Aggregate Vec Comm: %e\n", max_t);

                MPI_Reduce(&setup_mat_comm_times[2][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Aggregate Mat Comm: %e\n", max_t);

                MPI_Reduce(&setup_times[3][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Tent Interp: %e\n", max_t);

                MPI_Reduce(&setup_comm_times[3][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Tent Interp Vec Comm: %e\n", max_t);

                MPI_Reduce(&setup_mat_comm_times[3][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Tent Interp Mat Comm: %e\n", max_t);

                MPI_Reduce(&setup_times[4][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Prolongate: %e\n", max_t);

                MPI_Reduce(&setup_comm_times[4][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Prolongate Vec Comm: %e\n", max_t);

                MPI_Reduce(&setup_mat_comm_times[4][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Prolongate Mat Comm: %e\n", max_t);

                MPI_Reduce(&setup_times[5][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("A*P: %e\n", max_t);

                MPI_Reduce(&setup_comm_times[5][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("A*P Vec Comm: %e\n", max_t);

                MPI_Reduce(&setup_mat_comm_times[5][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("A*P Mat Comm: %e\n", max_t);

                MPI_Reduce(&setup_times[6][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("P.T*AP: %e\n", max_t);

                MPI_Reduce(&setup_comm_times[6][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("P.T*AP Vec Comm: %e\n", max_t);

                MPI_Reduce(&setup_mat_comm_times[6][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("P.T*AP Mat Comm: %e\n", max_t);

            }
        }


        agg_t agg_type;
        prolong_t prolong_type;
        aligned_vector<double> B;

        double interp_tol;
        double prolong_weight;
        int prolong_smooth_steps;
        int num_candidates;

    };
}
   

#endif




