#include "csrmatrix.h"

#include "block_partition.h"

namespace linalg
{
    namespace par
    {

        CSRMatrix::CSRMatrix(MPI_Comm comm,
                             sys::int_t numrows,
                             sys::int_t numcols,
                             Partition* rowstarts,
                             Partition* colstarts,
                             sys::int_t num_cols_offd,
                             sys::int_t nnz_diag,
                             sys::int_t nnz_offd):
            comm_(comm),
            global_num_rows_(numrows),
            global_num_cols_(numcols),
            row_part_(rowstarts),
            col_part_(colstarts)
        {
            int rank;
            MPI_Comm_rank(comm_, &rank);
            sys::int_t local_num_rows, local_num_cols;

            first_row_index = row_part_->low(rank);
            local_num_rows = row_part_->size(rank);
            first_col_diag = col_part_->low(rank);
            local_num_cols = col_part_->size(rank);

            diag_ = new seq::CSRMatrix(local_num_rows, local_num_cols,
                                      nnz_diag);
            offd_ = new seq::CSRMatrix(local_num_rows, num_cols_offd,
                                      nnz_offd);
            last_row_index = row_part_->high(rank);
            last_col_diag = col_part_->high(rank);
        }

        CSRMatrix::CSRMatrix(MPI_Comm comm,
                             sys::int_t numrows,
                             sys::int_t numcols,
                             sys::int_t num_cols_offd,
                             sys::int_t nnz_diag,
                             sys::int_t nnz_offd):
            comm_(comm),
            global_num_rows_(numrows),
            global_num_cols_(numcols)
        {
            int rank, commsize;
            MPI_Comm_rank(comm_, &rank);
            MPI_Comm_size(comm_, &commsize);
            sys::int_t local_num_rows, local_num_cols;

            row_part_ = new BlockPartition(numrows, commsize);
            col_part_ = new BlockPartition(numcols, commsize);

            first_row_index = row_part_->low(rank);
            local_num_rows = row_part_->size(rank);
            first_col_diag = col_part_->low(rank);
            local_num_cols = col_part_->size(rank);

            diag_ = new seq::CSRMatrix(local_num_rows, local_num_cols,
                                      nnz_diag);
            offd_ = new seq::CSRMatrix(local_num_rows, num_cols_offd,
                                      nnz_offd);
            last_row_index = row_part_->high(rank);
            last_col_diag = col_part_->high(rank);
        }

        CSRMatrix::~CSRMatrix()
        {
            delete row_part_;
            delete col_part_;
            delete diag_;
            delete offd_;
        }

        void CSRMatrix::local_range(sys::int_t *row_start,
                                        sys::int_t *row_end,
                                        sys::int_t *col_start,
                                        sys::int_t *col_end)
        {
            *row_start = first_row_index;
            *row_end = last_row_index;
            *col_start = first_col_diag;
            *col_end = last_col_diag;
        }

        void CSRMatrix::mult(Vector *invec, Vector *outvec)
        {

        }
    }
}
