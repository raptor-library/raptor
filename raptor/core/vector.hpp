// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef VECTOR_HPP
#define VECTOR_HPP

#include "types.hpp"
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <cmath>

namespace raptor
{
class Vector
{

public:
    Vector(index_t len)
    {
        values.resize(len);
        size = len;
    }

    Vector()
    {

    }

    void set_const_value(data_t alpha)
    {
        for (index_t i = 0; i < size; i++)
        {
            values[i] = alpha;
        }
    }

    void set_rand_values()
    {
        srand(time(NULL));
        for (index_t i = 0; i < size; i++)
        {
            values[i] = rand() / RAND_MAX;
        }
    }

    void axpy(Vector& x, data_t alpha)
    {
        for (index_t i = 0; i < size; i++)
        {
            values[i] += x.values[i]*alpha;
        }
    }

    void scale(data_t alpha)
    {
        for (index_t i = 0; i < size; i++)
        {
            values[i] *= alpha;
        }
    }

    data_t norm(index_t p)
    {
        data_t result = 0.0;
        for (index_t i = 0; i < size; i++)
        {
            result += pow(values[i], p);
        }
        return pow(result, 1.0/p);
    }

    data_t* data()
    {
        return values.data();
    }

    std::vector<data_t> values;
    index_t size;
};

}


#endif
