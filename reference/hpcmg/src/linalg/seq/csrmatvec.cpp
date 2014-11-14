#include "csrmatvec.h"
#include "sys/types.h"

namespace linalg
{
    namespace seq
    {
        void csr_matvec(const CSRMatrix &mat, const Vector &invec, Vector *outvec)
        {
            sys::int_t i, j;
            sys::int_t nrows = mat.num_rows();
            sys::data_t sum;
            sys::int_t curnnz;
            const Vector &rowptr = mat.rowptr();
            const Vector &colind = mat.colind();
            const Vector &data = mat.data();

            for (i=0;i<nrows;i++) {
                sum = 0.0;
                curnnz = mat.nnz(i);
                for (j=0;j<curnnz;j++) {
                    sum += data(rowptr(i) + j) *
                        invec(colind(colind(rowptr(i)+j)));
                }
                (*outvec)(i) = sum;
            }
        }
    }
}
