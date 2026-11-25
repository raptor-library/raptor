// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_PAR_HIP_MULTIVECTOR_HPP_
#define RAPTOR_CORE_PAR_HIP_MULTIVECTOR_HPP_

#include "hip_multivector.hpp"

namespace raptor
{
struct ParHipMultiVectorView
{
    HipMultiVectorView local;
    int global_rows = 0;
};
}

#endif
