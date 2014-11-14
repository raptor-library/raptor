#include "vector.h"

#include <cmath>

namespace linalg
{
    namespace seq
    {
        Vector::Vector(sys::int_t size)
            :len(size)
        {
            data = new sys::data_t[size];
        }


        Vector::~Vector()
        {
            delete[] data;
        }

        void Vector::scale(sys::data_t val)
        {
            int i;
            for (i=0;i<len;i++)
                data[i] = data[i]*val;
        }

	void Vector::set(sys::data_t val)
	{
	    sys::int_t i;
	    for (i=0;i<len;i++) {
		data[i] = val;
	    }
	}

        void Vector::abs()
        {
            int i;
            for (i=0;i<len;i++)
                data[i] = std::abs(data[i]);
        }


        sys::data_t Vector::dot(Vector* y)
        {
            // TODO: assert y.size() = self.size()
            sys::int_t i;
            sys::data_t retval = 0.0;
            for (i=0;i<len;i++)
                retval += data[i] * (*y)(i);
            return retval;
        }

        std::ostream &operator<<(std::ostream &os, const Vector & obj)
        {
            for (sys::int_t i=0; i < obj.len; i++) {
                os << obj.data[i] << ' ';
            }
            os << '\n';
            return os;
        }
    }
}
