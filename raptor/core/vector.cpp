// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "vector.hpp"

using namespace raptor;

    void Vector::set_const_value(data_t alpha)
    {
        for (index_t i = 0; i < size; i++)
        {
            values[i] = alpha;
        }
    }

    void Vector::set_rand_values()
    {
        srand(time(NULL));
        for (index_t i = 0; i < size; i++)
        {
            values[i] = rand() / RAND_MAX;
        }
    }

    void Vector::axpy(Vector& x, data_t alpha)
    {
        for (index_t i = 0; i < size; i++)
        {
            values[i] += x.values[i]*alpha;
        }
    }

    void Vector::scale(data_t alpha)
    {
        for (index_t i = 0; i < size; i++)
        {
            values[i] *= alpha;
        }
    }

    data_t Vector::norm(index_t p)
    {
        data_t result = 0.0;
        for (index_t i = 0; i < size; i++)
        {
            result += pow(values[i], p);
        }
        return pow(result, 1.0/p);
    }

    data_t* Vector::data()
    {
        return values.data();
    }


