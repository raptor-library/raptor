// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_PAR_SMOOTHED_AGGREGATION_SOLVER_HPP
#define RAPTOR_PAR_SMOOTHED_AGGREGATION_SOLVER_HPP

#include "raptor/multilevel/par_multilevel.hpp"
#include "par_mis.hpp"
#include "par_aggregate.hpp"
#include "par_candidates.hpp"
#include "par_prolongation.hpp"

#include <stdexcept>

namespace raptor
{
    template <class T, is_bsr_or_csr<T> = true>
    class ParSmoothedAggregationSolver_T : public ParMultilevel_T<T>
    {
      public:
        ParSmoothedAggregationSolver_T(double _strong_threshold = 0.0, 
                agg_t _agg_type = MIS, 
                prolong_t _prolong_type = default_prolong_type(),
                strength_t _strength_type = Classical,
                relax_t _relax_type = default_relax_type(),
                int _prolong_smooth_steps = 1, 
                double _prolong_weight = 4.0/3) 
            : ParMultilevel_T<T>(_strong_threshold, _strength_type, _relax_type)
        {
            wrong_type_throw_parSA(_prolong_type, _strength_type);
            agg_type = _agg_type;
            prolong_type = _prolong_type;
            num_candidates = 1;
            interp_tol = 1e-10;
            prolong_smooth_steps = _prolong_smooth_steps;
            prolong_weight = _prolong_weight;
        }

        ~ParSmoothedAggregationSolver_T()
        {
        }

        void setup(T* Af) override
        {
            // reject parbsrmatrix passed through parcsrmatrix 
            if constexpr (!is_bsr_v<T>)
            {
                if (Af->on_proc->format()==BSR)
                {
                    throw std::invalid_argument("Matrix is ParBSRMatrix; use ParSmoothedAggregationSolver_T<ParBSRMatrix>");
                }
            }
            // default B = constant vector 
            num_candidates = 1;
            B.assign(total_local_num_rows(*Af), 1.0);

            this->setup_helper(Af);
        }

        void setup(T* Af, const std::vector<double>& candidates, int num_cand)
        {
            // reject parbsrmatrix passed through parcsrmatrix 
            if constexpr (!is_bsr_v<T>)
            {
                if (Af->on_proc->format()==BSR)
                {
                    throw std::invalid_argument("Matrix is ParBSRMatrix; use ParSmoothedAggregationSolver_T<ParBSRMatrix>");
                }
            }
            assert(num_cand > 0);
            num_candidates = num_cand;
            B = candidates;

            this->setup_helper(Af);
        }

        void extend_hierarchy() override
        {
            int level_ctr = this->levels.size() - 1;
            bool tap_level = this->tap_amg >= 0 && this->tap_amg <= level_ctr;

            T* A = this->levels[level_ctr]->A;
            ParCSRMatrix* S;
            T* tentative;
            T* P = nullptr;
            T* AP;

            std::vector<int> states;
            std::vector<int> off_proc_states;
            std::vector<int> aggregates;
            std::vector<double> R;
            int n_aggs = 0;

            // Form strength of connection
            S = A->strength(this->strength_type, this->strong_threshold, tap_level, 
                    1, nullptr);

            // Aggregate Nodes
            switch (agg_type)
            {
                case MIS:
                    mis2(S, states, off_proc_states, tap_level, this->weights);
                    n_aggs = aggregate(A, S, states, off_proc_states, 
                            aggregates, tap_level);
                    break;
                default:
                    mis2(S, states, off_proc_states, tap_level, this->weights);
                    n_aggs = aggregate(A, S, states, off_proc_states, 
                            aggregates, tap_level);
                    break;
            }

            // Form tentative interpolation
            tentative = fit_candidates(A, n_aggs, aggregates, B, R, 
                    num_candidates, false, interp_tol);
            
            switch (prolong_type)
            {
                case JacobiProlongation:
                    P = jacobi_prolongation(A, tentative, tap_level, 
                            prolong_weight, prolong_smooth_steps);
                    break;
                case BlockJacobiProlongation:
                    P = jacobi_prolongation(A, tentative, tap_level, 
                            prolong_weight, prolong_smooth_steps, prolongation_weighting::block);
                    break;
                case BlockJacobiSpectralProlongation:
                    P = jacobi_prolongation(A, tentative, tap_level, 
                            prolong_weight, prolong_smooth_steps, prolongation_weighting::block_spectral);
                    break;
                default:
                    P = jacobi_prolongation(A, tentative, tap_level, 
                            prolong_weight, prolong_smooth_steps);
                    break;
            }
            this->levels[level_ctr]->P = P;

            // Form coarse grid operator
            this->levels.emplace_back(new ParLevel_T<T>());

            if constexpr (is_bsr_v<T>)
            {
                AP = A->mult(this->levels[level_ctr]->P);
            }
            else
            {
                AP = A->mult(this->levels[level_ctr]->P, tap_level);
            }

            if constexpr (is_bsr_v<T>)
            {
                A = AP->mult_T(P);
            }
            else
            {
                A = AP->mult_T(P, tap_level);
            }

            level_ctr++;
            this->levels[level_ctr]->A = A;
            A->comm = new ParComm(A->partition, A->off_proc_column_map,
                    A->on_proc_column_map, this->levels[level_ctr-1]->A->comm->key,
                    this->levels[level_ctr-1]->A->comm->mpi_comm);
            this->levels[level_ctr]->x.resize(total_global_num_rows(*A),total_local_num_rows(*A));
            this->levels[level_ctr]->b.resize(total_global_num_rows(*A),total_local_num_rows(*A));
            this->levels[level_ctr]->tmp.resize(total_global_num_rows(*A),total_local_num_rows(*A));
            this->levels[level_ctr]->P = nullptr;

            if (this->tap_amg >= 0 && this->tap_amg <= level_ctr)
            {
                // Create 2-step node-aware communicator for setup phase
                // will be changed to 3-step before solve phase
                this->levels[level_ctr]->A->tap_comm = new TAPComm(
                        this->levels[level_ctr]->A->partition,
                        this->levels[level_ctr]->A->off_proc_column_map,
                        this->levels[level_ctr]->A->on_proc_column_map, 
                        true, A->comm->mpi_comm);
            }

            update_B(B, R, num_candidates, total_local_num_rows(*A));

            delete AP;
            delete tentative;
            delete S;
        }    


        agg_t agg_type;
        prolong_t prolong_type;
        std::vector<double> B;

        double interp_tol;
        double prolong_weight;
        int prolong_smooth_steps;
        int num_candidates;

        private:
        static constexpr prolong_t default_prolong_type()
        {
            if constexpr(is_bsr_v<T>) return BlockJacobiProlongation;
            else return JacobiProlongation;
        }

        static constexpr relax_t default_relax_type()
        {
            if constexpr(is_bsr_v<T>) return BlockJacobi;
            else return SOR;
        }

        static void wrong_type_throw_parSA(prolong_t prolong_type, strength_t strength_type)
        {
            if constexpr(is_bsr_v<T>)
            {
                if (strength_type == Symmetric)
                {
                    throw std::invalid_argument("ParBSRMatrix doesn't support Symmetric strength yet");
                }

            }
            else
            {
                if (prolong_type == BlockJacobiProlongation)
                {
                    throw std::invalid_argument("ParCSRMatrix doesn't support BlockJacobiProlongation");
                }
                
            }
        }

        // update coarse grid Bc from R
        // R is row major, B is column major
        void update_B(std::vector<double>& candidates, const std::vector<double>& R, const int num_cand, const int total_local_num_rows)
        {
            assert(R.size() == total_local_num_rows*num_cand);
            std::vector<double> Bc(R.size(),0.0);
            for (int r = 0; r < total_local_num_rows; r++)
            {
                for (int c = 0; c < num_cand; c++)
                {
                    Bc[c*total_local_num_rows+r] = R[r*num_cand+c];
                }
            }
            candidates = std::move(Bc);
        }

    };

    using ParSmoothedAggregationSolver = ParSmoothedAggregationSolver_T<ParCSRMatrix>;
}
   

#endif
