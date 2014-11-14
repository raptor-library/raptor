#ifndef LINALG_SEQ_VECTOR_H
#define LINALG_SEQ_VECTOR_H

#include <iostream>

#include "sys/types.h"

namespace linalg
{
    namespace seq
    {
        /**
         * Sequential vector class.
         */
        class Vector
        {

            public:
                /**
                 * Constructs a sequential vector.
                 * @param size Length of vector to create.
                 */
                Vector(sys::int_t size);
                ~Vector();
                /**
                 * @return Length of the vector
                 */
                sys::int_t size() const { return len; };
                sys::data_t & operator () (sys::int_t i) { return data[i]; };
                const sys::data_t & operator () (sys::int_t i) const { return data[i]; };
                friend std::ostream &operator << (std::ostream &os, const Vector &obj);
                void scale(sys::data_t);
		void set(sys::data_t val);
                void abs();
                sys::data_t dot(Vector *y);

            private:
                sys::data_t *data;
                sys::int_t   len;
        };
    }
}
#endif
