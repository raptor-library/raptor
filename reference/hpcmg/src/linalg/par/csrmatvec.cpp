#include "csrmatvec.h"
#include "comm_pkg.h"
#include "commhandle.h"
#include "../seq/csrmatrix.h"

namespace linalg
{
    namespace par
    {
        // y <- A*x
        void csr_matvec(const CSRMatrix &A, const Vector &x, Vector *y)
        {
            CommPkg *comm_pkg = A.comm_pkg();
            const seq::CSRMatrix &diag = A.diag();
            const seq::CSRMatrix &offd = A.offd();
            const seq::Vector &x_local = x.local();
            seq::Vector &y_local = y->local();
        }
    }
}
