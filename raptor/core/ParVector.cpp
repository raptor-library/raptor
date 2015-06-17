/******** ParVector.cpp **********/
#include "ParVector.hpp"

using namespace raptor;

ParVector::ParVector(len_t gblN, len_t lclN):
	globalN(gblN), localN(lclN)
{
	local.resize(lclN);
}

ParVector::ParVector(ParVector&& v) = default;

template<int p>
data_t ParVector::norm() const
{
    data_t result = local.lpNorm<p>();

    result = pow(result, p); // undoing root of p from local operation
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return pow(result, 1./p);
}
