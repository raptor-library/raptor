// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef RAPTOR_PAR_AIR_SOLVER_HPP
#define RAPTOR_PAR_AIR_SOLVER_HPP

#include "raptor/multilevel/par_multilevel.hpp"

namespace raptor {

struct ParAIRSolver : ParMultilevel {

	ParAIRSolver(double strong_threshold_ = 0.0,
	             coarsen_t coarsen_type_ = RS,
	             interp_t interp_type_ = OnePoint,
	             strength_t strength_type_ = Classical,
	             restrict_t restrict_type_ = AIR,
	             relax_t relax_type_ = FCJacobi) :
		ParMultilevel(strong_threshold_, strength_type_, relax_type_),
		coarsen_type(coarsen_type_),
		interp_type(interp_type_),
		restrict_type(restrict_type_),
		variables(nullptr)
	{
		num_variables = 1;
	}

	void setup(ParCSRMatrix * Af) override {
		setup_helper(Af);
	}

	void extend_hierarchy() override;
private:
	template<class T>
	void extend_hier(T & A);

	void finalize_coarse_op(bool tap_init);

	void init_coarse_vectors();

	coarsen_t coarsen_type;
	interp_t interp_type;
	restrict_t restrict_type;
	int *variables;
};

}

#endif
