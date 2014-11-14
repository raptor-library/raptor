#ifndef LINALG_SEQ_CSRMATRIX_H
#define LINALG_SEQ_CSRMATRIX_H

#include <iostream>

#include "sys/types.h"
#include "vector.h"

namespace linalg
{
    namespace seq
    {
        /**
         * Compressed Sparse Row Matrix
         */
        class CSRMatrix
        {
            public:
                /**
                 * @param nrows Number of rows
                 * @param nnz Number of Nonzeros
                 */
                CSRMatrix(sys::int_t nrows, sys::int_t ncols,
                          sys::int_t nnz);
                ~CSRMatrix();
                Vector &data() { return *data_; };
                Vector &rowptr() { return *rowptr_; };
                Vector &colind() { return *colind_; };
                const Vector &data() const { return *data_; };
                const Vector &rowptr() const { return *rowptr_; };
                const Vector &colind() const { return *colind_; };
                sys::int_t num_rows() const { return num_rows_; };
                sys::int_t num_cols() const { return num_cols_; };
                sys::int_t nnz() const { return nnz_; };
                sys::int_t &nnz() { return nnz_; };
                sys::int_t  nnz(sys::int_t i) const { return (*rowptr_)(i+1) - (*rowptr_)(i); };
                void scale(sys::data_t);
                void mult(const Vector &invec, Vector *outvec) const;
                friend std::ostream &operator << (std::ostream &os, const CSRMatrix &obj);

            private:
                Vector      *data_;
                Vector      *rowptr_;
                Vector      *colind_;
                sys::int_t   num_rows_;
                sys::int_t   num_cols_;
                sys::int_t   nnz_;
                bool         ownsdata;
        };
    }
}
#endif
