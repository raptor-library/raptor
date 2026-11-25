// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_HIP_MULTIVECTOR_HPP_
#define RAPTOR_CORE_HIP_MULTIVECTOR_HPP_

namespace raptor
{
struct HipMultiVectorView
{
    double* data = nullptr;
    int rows = 0;
    int cols = 0;
};
}

#endif
