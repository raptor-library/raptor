// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_PARVECTOR_HPP
#define RAPTOR_CORE_PARVECTOR_HPP

#include <mpi.h>
#include <math.h>

#include "Types.hpp"
#include "Vector.hpp"

namespace raptor
{

    class ParVector
    {
    public:
        ParVector(index_t N, index_t n);
        ParVector(ParVector&& x);
        ~ParVector() {};

        template<int p> data_t norm() const
        {
            data_t result = local.lpNorm<p>();

            result = pow(result, p); // undoing root of p from local operation
            MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            return pow(result, 1./p);
        }

        Vector & getLocalVector()
        {
            return local;
        }
        const Vector & getLocalVector() const
        {
            return local;
        }
        void axpy(const ParVector & x, data_t alpha)
        {
            local += x.getLocalVector() * alpha;
        }
        void scale(data_t alpha)
        {
            local *= alpha;
        }
        void setConstValue(data_t alpha)
        {
            local = Vector::Constant(localN, alpha);
        }

    protected:
        Vector local;
        int globalN;
        int localN;
    };

}
#endif
