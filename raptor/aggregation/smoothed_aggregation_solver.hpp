// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_SMOOTHED_AGGREGATION_SOLVER_HPP
#define RAPTOR_SMOOTHED_AGGREGATION_SOLVER_HPP

#include "multilevel/multilevel.hpp"
#include "aggregation/mis.hpp"
#include "aggregation/aggregate.hpp"
#include "aggregation/candidates.hpp"
#include "aggregation/prolongation.hpp"

namespace raptor
{
    class SmoothedAggregationSolver : public Multilevel
    {
      public:
        SmoothedAggregationSolver(double _strong_threshold = 0.0, agg_t _agg_type = MIS, 
                prolong_t _prolong_type = JacobiProlongation,
                strength_t _strength_type = Symmetric,
                relax_t _relax_type = SOR) 
            : Multilevel(_strong_threshold, _strength_type, _relax_type)
        {
            agg_type = _agg_type;
            prolong_type = _prolong_type;
            num_candidates = 1;
            interp_tol = 1e-10;
            prolong_smooth_steps = 1;
            prolong_weight = 4.0/3;
        }

        ~SmoothedAggregationSolver()
        {

        }

        void setup(CSRMatrix* Af) 
        {
            // TODO -- add option for B to be passed as variable
            num_candidates = 1;
            B.resize(Af->n_rows);
            for (int i = 0; i < Af->n_rows; i++)
            {
                B[i] = 1.0;
            }

            setup_helper(Af);
        }

        void extend_hierarchy()
        {
            int level_ctr = levels.size() - 1;
            CSRMatrix* A = levels[level_ctr]->A;
            CSRMatrix* S;
            CSRMatrix* P;
            CSRMatrix* AP;
            CSRMatrix* T;
            CSCMatrix* P_csc;
            aligned_vector<int> states;
            aligned_vector<int> aggregates;
            aligned_vector<double> R;
            int n_aggs;

            // Form Strength Matrix
            printf("Forming S..\n");
            S = A->strength(strength_type, strong_threshold);

            // Aggregate Nodes
            switch (agg_type)
            {
                case MIS:
                    mis2(S, states, weights);
                    n_aggs = aggregate(A, S, states, aggregates);
                    break;
            }
            
            // Form tentative interpolation
            T = fit_candidates(n_aggs, aggregates, B, R, num_candidates, interp_tol);
            
            switch (prolong_type)
            {
                case JacobiProlongation:
                    P = jacobi_prolongation(A, T, prolong_weight, prolong_smooth_steps);
                    break;
            }
            levels[level_ctr]->P = P;

            // Form coarse-grid operator
            levels.push_back(new Level());
            AP = A->mult(P);
            P_csc = P->to_CSC();
            A = AP->mult_T(P_csc);

            level_ctr++;
            levels[level_ctr]->A = A;
            levels[level_ctr]->x.resize(A->n_rows);
            levels[level_ctr]->b.resize(A->n_rows);
            levels[level_ctr]->tmp.resize(A->n_rows);
            levels[level_ctr]->P = NULL;

            std::copy(R.begin(), R.end(), B.begin());

            delete T;
            delete AP;
            delete P_csc;
            delete S;
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


