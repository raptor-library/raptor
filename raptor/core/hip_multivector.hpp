// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_HIP_MULTIVECTOR_HPP_
#define RAPTOR_CORE_HIP_MULTIVECTOR_HPP_

namespace raptor
{
/**************************************************************
 *****   HIP Multivector View
 **************************************************************
 ***** This struct views caller-owned device storage in packed
 ***** column-major order.
 *****
 ***** Attributes
 ***** -------------
 ***** data : double*
 *****    Pointer to the first element of the device storage
 ***** rows : int
 *****    Number of scalar rows and the leading dimension
 ***** cols : int
 *****    Number of columns
 *****
 **************************************************************/
struct HipMultiVectorView
{
    double* data = nullptr;
    int rows = 0;
    int cols = 0;
};
}

#endif
