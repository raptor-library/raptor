// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_PAR_HIP_MULTIVECTOR_HPP_
#define RAPTOR_CORE_PAR_HIP_MULTIVECTOR_HPP_

#include "hip_multivector.hpp"
#include <stdexcept>

namespace raptor
{
/**************************************************************
 *****   Parallel HIP Multivector View
 **************************************************************
 ***** This struct views the rows of a distributed multivector
 ***** stored on the current rank's device.
 *****
 ***** Attributes
 ***** -------------
 ***** local : HipMultiVectorView
 *****    View of the rank's local rows for every column
 ***** global_rows : int
 *****    Total number of scalar rows across all ranks
 *****
 **************************************************************/
struct ParHipMultiVectorView
{
    HipMultiVectorView local;
    int global_rows = 0;
    ParHipMultiVectorView() = default;

    ParHipMultiVectorView(HipMultiVectorView local_view, int total_rows)
        : local(local_view), global_rows(total_rows)
    {
        if (local.rows < 0 || global_rows < local.rows)
            throw std::invalid_argument("Invalid multivector row counts");
    }
};
}

#endif
