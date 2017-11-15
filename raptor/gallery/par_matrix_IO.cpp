// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "par_matrix_IO.hpp"
#include <float.h>
#include <stdio.h>

ParCSRMatrix* readParMatrix(char* filename, MPI_Comm comm, bool single_file, 
        int symmetric, int local_num_rows, int local_num_cols, 
        int first_local_row, int first_local_col)
{
    index_t num_rows, num_cols, nnz;
    int comm_size, rank, ret_code;
    index_t* row_ptr;
    index_t* col;
    data_t* data;
    ParCOOMatrix* A;
    
    if (single_file) 
    {
        MM_typecode matcode;
        FILE* infile;
        
        MPI_Comm_size(comm, &comm_size);
        MPI_Comm_rank(comm, &rank);

        // find size of matix 
        if ((infile = fopen(filename, "r")) == NULL) 
            return NULL;

        if (mm_read_banner(infile, &matcode) != 0)
            return NULL;

        if ((ret_code = mm_read_mtx_crd_size(infile, &num_rows, &num_cols, &nnz)) !=0)
            return NULL;
        
        fclose(infile);

        if (first_local_col >= 0)
        {
            A = new ParCOOMatrix(num_rows, num_cols, local_num_rows, local_num_cols,
                    first_local_row, first_local_col);
        }
        else
        {
            A = new ParCOOMatrix(num_rows, num_cols);
        }

        // read the file knowing our local rows
        ret_code = mm_read_par_sparse(filename, A->partition->first_local_row,
            A->partition->first_local_row + A->partition->local_num_rows, &num_rows, &num_cols,
            A, symmetric);

        if (ret_code != 0)
        {
            return NULL;
        }
    }

    A->finalize();
    ParCSRMatrix* A_csr = new ParCSRMatrix(A);
    delete A;

    return A_csr;
}

int mm_read_par_sparse(const char *fname, index_t start, index_t stop, 
        index_t *M_, index_t *N_, ParMatrix* A, int symmetric)
{
    FILE *f;
    MM_typecode matcode;
    index_t M, N, nz;
    int i, ctr;
     
    if ((f = fopen(fname, "r")) == NULL)
            return -1;
 
 
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("mm_read_unsymetric: Could not process Matrix Market banner ");
        printf(" in file [%s]\n", fname);
        return -1;
    } 
 
    if ( mm_is_complex(matcode) || !(mm_is_matrix(matcode) &&
            mm_is_sparse(matcode)))
    {
        fprintf(stderr, "Sorry, this application does not support matrix type");
        return -1;
    }
 
    /* find out size of sparse matrix: M, N, nz .... */
 
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) !=0)
    {
        fprintf(stderr, "read_unsymmetric_sparse(): could not parse matrix size.\n");
        return -1;
    }
 
    *M_ = M;
    *N_ = N;
 
    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    if (mm_is_integer(matcode) || mm_is_real(matcode))
    {
        char buf[sizeof(index_t)*2 + sizeof(double) + sizeof(char)];

        for (i=0; i<nz; i++)
        {
            int row, col;
            double value;
            fscanf(f, "%d %d %lg\n", &row, &col, &value);

            if (fabs(value) < 1e-15) continue;

            if (row > start && row <= stop)
            {
                A->add_global_value(row-1, col-1, value);
            }
            if (symmetric && (col > start && col <= stop && col != row))
            {
                A->add_global_value(col-1, row-1, value);
            }
        }
    }


    fclose(f);

    return 0;
}

