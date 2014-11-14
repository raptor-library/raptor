#include "csrmatrix.h"
#include "csrmatvec.h"


namespace linalg
{
    namespace seq
    {
        CSRMatrix::CSRMatrix(sys::int_t nrows, sys::int_t ncols,
                             sys::int_t nnz):
            num_rows_(nrows),
            num_cols_(ncols),
            nnz_(nnz),
            ownsdata(true)
        {
            data_ = new Vector(nnz_);
            colind_ = new Vector(nnz_);
            rowptr_ = new Vector(num_rows_+1);
        }


        CSRMatrix::~CSRMatrix()
        {
            if (ownsdata) {
                delete data_;
                delete colind_;
                delete rowptr_;
            }
        }


        void CSRMatrix::scale(sys::data_t val)
        {
            sys::int_t i;
            for (i=0;i<nnz_;i++)
                (*data_)(i) = val;
        }


        void CSRMatrix::mult(const Vector &invec, Vector *outvec) const
        {
            csr_matvec(*this, invec, outvec);
        }


        std::ostream &operator<<(std::ostream &os, const CSRMatrix &obj)
        {
            sys::int_t i, j, curnnz;
            Vector &rowptr = *obj.rowptr_;
            Vector &colind = *obj.colind_;
            Vector &data = *obj.data_;

            os << "CSRMatrix\n";
            os << "=========\n";

            for (i=0; i < obj.num_rows_; i++) {
                curnnz = rowptr(i+1) - rowptr(i);
                for (j=0; j < curnnz; j++) {
                    os << "(" << std::to_string(i) << ", " 
                        << std::to_string(colind(rowptr(i)+j))
                        << ") -> " << data(rowptr(i)+j) << '\n';
                }
            }
            return os;
        }
    }
}
