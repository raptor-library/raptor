#ifndef LINALG_SEQ_MATVEC_H
#define LINALG_SEQ_MATVEC_H

#include "csrmatrix.h"
#include "vector.h"

namespace linalg
{
    namespace seq
    {
        void csr_matvec(const CSRMatrix &mat, const Vector &invec, Vector *outvec);
    }
}

#endif
