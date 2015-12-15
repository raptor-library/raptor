// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_vector.hpp"

using namespace raptor;

ParVector::ParVector(ParVector&& v) = default;

void ParVector::axpy(const ParVector & x, data_t alpha)
{
    if (local_n)
    {
        *local += *(x.local) * alpha;
    }
}
void ParVector::scale(data_t alpha)
{
    if (local_n)
    {
        *local *= alpha;
    }
}
void ParVector::set_const_value(data_t alpha)
{
    if (local_n)
    {
        *local = Vector::Constant(local_n, alpha);
    }
}
void ParVector::set_rand_values()
{
    if (local_n)
    {
        *local = Vector::Random(local_n);
    }
}
