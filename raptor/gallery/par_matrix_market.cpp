/* 
*   Matrix Market I/O library for ANSI C
*
*   See http://math.nist.gov/MatrixMarket for details.
*
*
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#include "par_matrix_market.hpp"


using namespace raptor;

ParCSRMatrix* read_par_mm(const char *fname)
{
    FILE *f;
    MM_typecode matcode;
    int M, N, nz;
    int i;
    int row, col;
    double val;
 
    if ((f = fopen(fname, "r")) == NULL)
            return NULL;
 
 
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("mm_read_unsymetric: Could not process Matrix Market banner ");
        printf(" in file [%s]\n", fname);
        return NULL;
    }
 
 
    if ( !(mm_is_real(matcode) && mm_is_matrix(matcode) &&
            mm_is_sparse(matcode)))
    {
        fprintf(stderr, "Sorry, this application does not support ");
        fprintf(stderr, "Market Market type: [%s]\n",
                mm_typecode_to_str(matcode));
        return NULL;
    }
 
    /* find out size of sparse matrix: M, N, nz .... */
 
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) !=0)
    {
        fprintf(stderr, "read_unsymmetric_sparse(): could not parse matrix size.\n");
        return NULL;
    }
 
    int row_nnz = nz / M;
    ParCOOMatrix* A = new ParCOOMatrix(M, N);
    A->on_proc->vals.reserve(row_nnz);
    A->off_proc->vals.reserve(row_nnz);

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
 
    bool symmetric = mm_is_symmetric(matcode);
    bool row_local;
    bool col_local;
    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &row, &col, &val);
        row--;
        col--;
        if (row >= A->partition->first_local_row && row <= A->partition->last_local_row)
        {
            row_local = true;
            row -= A->partition->first_local_row;
        }
        else
        {
            row_local = false;
            if (!symmetric) 
                continue;
        }
        if (col >= A->partition->first_local_col && col <= A->partition->last_local_col)
        {
            col_local = true;
            col -= A->partition->first_local_col;
        }
        else
        {
            col_local = false;
            if (!row_local) 
                continue;
        }

        if (row_local)
        {
            if (col_local)
            {
                A->on_proc->add_value(row, col, val);
            }
            else
            {
                A->off_proc->add_value(row, col, val);
            }
        }

        if (symmetric)
        {
            if (col_local)
            {
                if (row_local)
                {
                    A->on_proc->add_value(col, row, val);
                }
                else
                {
                    A->off_proc->add_value(col, row, val);
                }
            }
        }
    }

    A->finalize();
    ParCSRMatrix* A_csr = A->to_ParCSR();
    delete A;

    fclose(f);
 
    return A_csr;
}

void write_par_data(FILE* f, int n, int* rowptr, int* col_idx,
        double* vals, int first_row, int* col_map)
{
    int start, end, global_row;

    for (int i = 0; i < n; i++)
    {
        global_row = first_row + i;
        start = rowptr[i];
        end = rowptr[i+1];
        for (int j = start; j < end; j++)
        {
            fprintf(f, "%d %d %2.15e\n", global_row + 1, 
                    col_map[col_idx[j]] + 1, vals[j]);
        }
    }
}


void write_par_mm(ParCSRMatrix* A, const char *fname)
{
    int rank, num_procs;
    RAPtor_MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FILE *f;
    MM_typecode matcode;
    int pos, bytes;
    int int_bytes, double_bytes;
    int num_ints, num_doubles;
    int comm_size;

    aligned_vector<char> buffer;

    int nnz = A->local_nnz;
    int global_nnz;
    RAPtor_MPI_Reduce(&nnz, &global_nnz, 1, RAPtor_MPI_INT, RAPtor_MPI_SUM, 0, RAPtor_MPI_COMM_WORLD);

    std::vector<int> proc_dims(5*num_procs);
    int dims[5];
    dims[0] = A->local_num_rows + 1;
    dims[1] = A->on_proc_num_cols;
    dims[2] = A->off_proc_num_cols; 
    dims[3] = A->on_proc->nnz;
    dims[4] = A->off_proc->nnz;
    RAPtor_MPI_Gather(dims, 5, RAPtor_MPI_INT, proc_dims.data(), 5, RAPtor_MPI_INT, 0, RAPtor_MPI_COMM_WORLD);

    if (rank == 0) // RANK 0 IS ONLY ONE WRITING TO FILE
    {
        f = fopen(fname, "w");

        mm_initialize_typecode(&matcode);
        mm_set_matrix(&matcode);
        mm_set_coordinate(&matcode);
        mm_set_real(&matcode);

        mm_write_banner(f, matcode);
        fprintf(f, "%%\n");
        mm_write_mtx_crd_size(f, A->global_num_rows, A->global_num_cols,
                global_nnz);

        // Write local data
        int first_row = 0;
        write_par_data(f, A->local_num_rows, A->on_proc->idx1.data(),
                A->on_proc->idx2.data(), A->on_proc->vals.data(),
                first_row, A->on_proc_column_map.data());
        write_par_data(f, A->local_num_rows, A->off_proc->idx1.data(),
                A->off_proc->idx2.data(), A->off_proc->vals.data(),
                first_row, A->off_proc_column_map.data());
        first_row += A->local_num_rows;

        // Write data from other processes
        aligned_vector<int> idx1;
        aligned_vector<int> idx2;
        aligned_vector<double> vals;
        aligned_vector<int> row_map;
        aligned_vector<int> col_map; 
        for (int i = 1; i < num_procs; i++)
        {
            // Calculate comm_size and allocate recv_buf
            int* i_dims = &proc_dims[i*5];
            num_ints = i_dims[0] * 2 + i_dims[1] + i_dims[3] + i_dims[3] + i_dims[4];
            num_doubles = i_dims[3] + i_dims[4];
            RAPtor_MPI_Pack_size(num_ints, RAPtor_MPI_INT, RAPtor_MPI_COMM_WORLD, &int_bytes);
            RAPtor_MPI_Pack_size(num_doubles, RAPtor_MPI_DOUBLE, RAPtor_MPI_COMM_WORLD, &double_bytes);
            comm_size = int_bytes + double_bytes;
            if (buffer.size() < comm_size) buffer.resize(comm_size);

            // Resize Matrix Arrays
            int row_max = i_dims[0];
            int col_max = i_dims[1];
            int nnz_max = i_dims[3];
            if (i_dims[2] > i_dims[1]) col_max = i_dims[2];
            if (i_dims[4] > i_dims[3]) nnz_max = i_dims[4];
            if (col_map.size() < col_max) col_map.resize(col_max);
            if (idx1.size() < row_max) idx1.resize(row_max);
            if (idx2.size() < nnz_max)
            {
                idx2.resize(nnz_max);
                vals.resize(nnz_max);
            }

            // Recv Packed Buffer
            RAPtor_MPI_Recv(buffer.data(), comm_size, RAPtor_MPI_PACKED, i, 1234, RAPtor_MPI_COMM_WORLD,
                    RAPtor_MPI_STATUS_IGNORE);

            // Unpack On Proc Data
            pos = 0;
            RAPtor_MPI_Unpack(buffer.data(), comm_size, &pos, col_map.data(), i_dims[1],
                    RAPtor_MPI_INT, RAPtor_MPI_COMM_WORLD);
            RAPtor_MPI_Unpack(buffer.data(), comm_size, &pos, idx1.data(), i_dims[0],
                    RAPtor_MPI_INT, RAPtor_MPI_COMM_WORLD);
            RAPtor_MPI_Unpack(buffer.data(), comm_size, &pos, idx2.data(), i_dims[3],
                    RAPtor_MPI_INT, RAPtor_MPI_COMM_WORLD);
            RAPtor_MPI_Unpack(buffer.data(), comm_size, &pos, vals.data(), i_dims[3],
                    RAPtor_MPI_DOUBLE, RAPtor_MPI_COMM_WORLD);
            write_par_data(f, i_dims[0] - 1, idx1.data(), idx2.data(), 
                    vals.data(), first_row, col_map.data());

            RAPtor_MPI_Unpack(buffer.data(), comm_size, &pos, col_map.data(), i_dims[2],
                    RAPtor_MPI_INT, RAPtor_MPI_COMM_WORLD);
            RAPtor_MPI_Unpack(buffer.data(), comm_size, &pos, idx1.data(), i_dims[0],
                    RAPtor_MPI_INT, RAPtor_MPI_COMM_WORLD);
            RAPtor_MPI_Unpack(buffer.data(), comm_size, &pos, idx2.data(), i_dims[4],
                    RAPtor_MPI_INT, RAPtor_MPI_COMM_WORLD);
            RAPtor_MPI_Unpack(buffer.data(), comm_size, &pos, vals.data(), i_dims[4],
                    RAPtor_MPI_DOUBLE, RAPtor_MPI_COMM_WORLD);
            write_par_data(f, i_dims[0] - 1, idx1.data(), idx2.data(), 
                    vals.data(), first_row, col_map.data());

            first_row += i_dims[0] - 1;
        }

        fclose(f);
    }
    else // All processes that are not 0, send to 0
    {
        // Determine send size (in bytes)
        num_ints = dims[0] * 2 + dims[1] + dims[3] + dims[3] + dims[4];
        num_doubles = dims[3] + dims[4];
        RAPtor_MPI_Pack_size(num_ints, RAPtor_MPI_INT, RAPtor_MPI_COMM_WORLD, &int_bytes);
        RAPtor_MPI_Pack_size(num_doubles, RAPtor_MPI_DOUBLE, RAPtor_MPI_COMM_WORLD, &double_bytes);
        comm_size = int_bytes + double_bytes;
        buffer.resize(comm_size);

        // Pack Data
        pos = 0;
        RAPtor_MPI_Pack(A->on_proc_column_map.data(), dims[1], RAPtor_MPI_INT, buffer.data(), comm_size, 
               &pos, RAPtor_MPI_COMM_WORLD); 
        RAPtor_MPI_Pack(A->on_proc->idx1.data(), dims[0], RAPtor_MPI_INT, buffer.data(), comm_size,
                &pos, RAPtor_MPI_COMM_WORLD);
        RAPtor_MPI_Pack(A->on_proc->idx2.data(), dims[3], RAPtor_MPI_INT, buffer.data(), comm_size,
                &pos, RAPtor_MPI_COMM_WORLD);
        RAPtor_MPI_Pack(A->on_proc->vals.data(), dims[3], RAPtor_MPI_DOUBLE, buffer.data(), comm_size,
                &pos, RAPtor_MPI_COMM_WORLD);
        
        RAPtor_MPI_Pack(A->off_proc_column_map.data(), dims[2], RAPtor_MPI_INT, buffer.data(), comm_size, 
               &pos, RAPtor_MPI_COMM_WORLD); 
        RAPtor_MPI_Pack(A->off_proc->idx1.data(), dims[0], RAPtor_MPI_INT, buffer.data(), comm_size,
                &pos, RAPtor_MPI_COMM_WORLD);
        RAPtor_MPI_Pack(A->off_proc->idx2.data(), dims[4], RAPtor_MPI_INT, buffer.data(), comm_size,
                &pos, RAPtor_MPI_COMM_WORLD);
        RAPtor_MPI_Pack(A->off_proc->vals.data(), dims[4], RAPtor_MPI_DOUBLE, buffer.data(), comm_size,
                &pos, RAPtor_MPI_COMM_WORLD);

        // Send Packed Data
        RAPtor_MPI_Send(buffer.data(), comm_size, RAPtor_MPI_PACKED, 0, 1234, RAPtor_MPI_COMM_WORLD);
    }
}

