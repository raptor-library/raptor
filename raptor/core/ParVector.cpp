// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "ParVector.hpp"

using namespace raptor;

ParVector::ParVector(len_t gblN, len_t lclN):
    globalN(gblN), localN(lclN)
{
    local.resize(lclN);
}

ParVector::ParVector(ParVector&& v) = default;
