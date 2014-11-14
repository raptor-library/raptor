#ifndef LINALG_PAR_CSRMATRIX_H
#define LINALG_PAR_CSRMATRIX_H

#include <mpi.h>

#include "linalg/seq/csrmatrix.h"
#include "sys/types.h"
#include "partition.h"
#include "vector.h"
#include "comm_pkg.h"

namespace linalg
{
    namespace par
    {
        class CommPkg;
        /**
         * Parallel CSR Matrix
         */
        class CSRMatrix{

            public:
                CSRMatrix(MPI_Comm comm,
                          sys::int_t numrows,
                          sys::int_t numcols,
                          Partition *rowstarts,
                          Partition *colstarts,
                          sys::int_t num_cols_offd,
                          sys::int_t nnz_diag,
                          sys::int_t nnz_offd);
                CSRMatrix(MPI_Comm comm,
                          sys::int_t numrows,
                          sys::int_t numcols,
                          sys::int_t num_cols_offd,
                          sys::int_t nnz_diag,
                          sys::int_t nnz_offd);
                ~CSRMatrix();

                CommPkg *comm_pkg() const { return comm_pkg_; };
                seq::CSRMatrix &diag() { return *diag_; };
                seq::CSRMatrix &offd() { return *offd_; };
                const seq::CSRMatrix &diag() const { return *diag_; };
                const seq::CSRMatrix &offd() const { return *offd_; };
                MPI_Comm comm() { return comm_; };
                sys::int_t num_cols_diag() { return last_col_diag - first_col_diag; };
                sys::int_t *col_map_offd() { return col_map_offd_; };
                Partition *col_part() { return col_part_; };
                Partition *row_part() { return row_part_; };
                sys::int_t global_num_rows() const { return global_num_rows_; };
                sys::int_t global_num_cols() const { return global_num_cols_; };

                void local_range(sys::int_t *row_start,
                                 sys::int_t *row_end,
                                 sys::int_t *col_start,
                                 sys::int_t *col_end);

                void mult(Vector *invec, Vector *outvec);

            private:
                MPI_Comm comm_;
                CommPkg *comm_pkg_;

                seq::CSRMatrix *diag_;
                seq::CSRMatrix *offd_;

                sys::int_t global_num_rows_;
                sys::int_t global_num_cols_;
                sys::int_t first_row_index;
                sys::int_t first_col_diag;
                sys::int_t last_row_index;
                sys::int_t last_col_diag;

                sys::int_t *col_map_offd_;
                Partition  *row_part_;
                Partition  *col_part_;

                sys::int_t nnz;
        };
    }
}
#endif
