// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "ParVector.hpp"

//using namespace raptor;

ParVector::ParVector(index_t gbl_n, index_t lcl_n):
    global_n(gblN), local_n(lcl_n)
{
    local.resize(lcl_n);
}

ParVector::ParVector(ParVector&& v) = default;
