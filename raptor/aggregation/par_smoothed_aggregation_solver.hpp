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
                relax_t _relax_type = SOR) 
            : ParMultilevel(_strong_threshold, _strength_type, _relax_type)
        {
            agg_type = _agg_type;
            prolong_type = _prolong_type;
            num_candidates = 1;
            interp_tol = 1e-10;
            prolong_smooth_steps = 1;
            prolong_weight = 4.0/3;
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

            if (setup_times) setup_times[0][level_ctr] -= MPI_Wtime();

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
            S = A->strength(strength_type, strong_threshold);
            if (setup_times) setup_times[1][level_ctr] += MPI_Wtime();

            // Aggregate Nodes
            if (setup_times) setup_times[2][level_ctr] -= MPI_Wtime();
            switch (agg_type)
            {
                case MIS:
                    mis2(S, states, off_proc_states, weights);
                    n_aggs = aggregate(A, S, states, off_proc_states, aggregates);
                    break;
            }
            if (setup_times) setup_times[2][level_ctr] += MPI_Wtime();

            // Form modified classical interpolation
            if (setup_times) setup_times[3][level_ctr] -= MPI_Wtime();
            // Form tentative interpolation
            T = fit_candidates(A, n_aggs, aggregates, B, R, num_candidates, interp_tol);
            if (setup_times) setup_times[3][level_ctr] += MPI_Wtime();
            

            if (setup_times) setup_times[4][level_ctr] -= MPI_Wtime();
            switch (prolong_type)
            {
                case JacobiProlongation:
                    P = jacobi_prolongation(A, T, prolong_weight, prolong_smooth_steps);
                    break;
            }
            if (setup_times) setup_times[4][level_ctr] += MPI_Wtime();
            levels[level_ctr]->P = P;

            // Form coarse grid operator
            levels.push_back(new ParLevel());

            if (tap_level)
            {
                if (setup_times) setup_times[5][level_ctr] -= MPI_Wtime();
                AP = A->tap_mult(levels[level_ctr]->P);
                if (setup_times) setup_times[5][level_ctr] += MPI_Wtime();


                if (setup_times) setup_times[6][level_ctr] -= MPI_Wtime();
                A = AP->tap_mult_T(P);
                if (setup_times) setup_times[6][level_ctr] += MPI_Wtime();
            }
            else
            {
                if (setup_times) setup_times[5][level_ctr] -= MPI_Wtime();
                AP = A->mult(levels[level_ctr]->P);
                if (setup_times) setup_times[5][level_ctr] += MPI_Wtime();

                if (setup_times) setup_times[6][level_ctr] -= MPI_Wtime();
                A = AP->mult_T(P);
                if (setup_times) setup_times[6][level_ctr] += MPI_Wtime();
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

            std::copy(R.begin(), R.end(), B.begin());

            delete T;
            delete S;

            if (setup_times) setup_times[0][level_ctr-1] += MPI_Wtime();
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

                MPI_Reduce(&setup_times[1][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Strength: %e\n", max_t);

                MPI_Reduce(&setup_times[2][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Aggregate: %e\n", max_t);

                MPI_Reduce(&setup_times[3][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Tent Interp: %e\n", max_t);

                MPI_Reduce(&setup_times[4][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("Prolongate: %e\n", max_t);

                MPI_Reduce(&setup_times[5][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("A*P: %e\n", max_t);

                MPI_Reduce(&setup_times[6][i], &max_t, 1, MPI_DOUBLE, 
                        MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && max_t > 0) printf("P.T*AP: %e\n", max_t);

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




