#ifndef PAR_CSRMATVEC_H
#define PAR_CSRMATVEC_H

#include "csrmatrix.h"
#include "vector.h"

namespace linalg
{
    namespace par
    {
        void csr_matvec(const CSRMatrix &mat, const Vector &invec, Vector *outvec);
    }
}

#endif
