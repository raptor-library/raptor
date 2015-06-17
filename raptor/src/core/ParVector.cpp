/******** ParVector.cpp **********/
#include "ParVector.hpp"

using namespace raptor;

ParVector::ParVector(len_t gblN, len_t lclN):
	globalN(gblN), localN(lclN)
{
	local.resize(lclN);
}

ParVector::ParVector(ParVector&& v) = default;
