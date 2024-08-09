// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef RAPTOR_PAR_AIR_SOLVER_HPP
#define RAPTOR_PAR_AIR_SOLVER_HPP

#include "raptor/multilevel/par_multilevel.hpp"

namespace raptor {

struct ParAIRSolver : ParMultilevel {

	ParAIRSolver(double strong_threshold = 0.0,
	             coarsen_t coarsen_type = RS,
	             interp_t interp_type = OnePoint,
	             strength_t strength_type = Classical,
	             restrict_t restrict_type = AIR,
	             relax_t relax_type = FCJacobi) :
		ParMultilevel(strong_threshold, strength_type, relax_type),
		coarsen_type(coarsen_type),
		interp_type(interp_type),
		restrict_type(restrict_type),
		variables(nullptr)
	{
		num_variables = 1;
	}

	void setup(ParCSRMatrix * Af) override {
		setup_helper(Af);
	}

	void extend_hierarchy() override {
		int level_ctr = levels.size() - 1;
		ParCSRMatrix *A = levels[level_ctr]->A;
		ParBSRMatrix *A_bsr = dynamic_cast<ParBSRMatrix*>(A);
		if (A_bsr) extend_hier(*A_bsr);
		// else extend_hier(*A);
	}

private:
	template<class T>
	void extend_hier(T & A) {
		int level_ctr = levels.size() - 1;
		bool tap_level = tap_amg >= 0 && tap_amg <= level_ctr;
		splitting_t split;
		auto S = A.strength(strength_type, strong_threshold, tap_level,
		                    num_variables, variables);

		split_rs(S, split.on_proc, split.off_proc, tap_level);
		auto P = one_point_interpolation(A, *S, split);
		auto R = local_air(A, *S, split, fpoint_distance::two);

		levels.emplace_back(new ParLevel());
	}

	coarsen_t coarsen_type;
	interp_t interp_type;
	restrict_t restrict_type;
	int *variables;
};

}

#endif
