#include "matrix_IO.hpp"
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
        ret_code = mm_read_sparse(filename, A->first_local_row,
            A->first_local_row + A->local_num_rows, &num_rows, &num_cols,
            A, symmetric);

        if (ret_code != 0)
        {
            return NULL;
        }
    }

    ParCSRMatrix* A_csr = new ParCSRMatrix(A);
    A_csr->finalize();
    delete A;

    return A_csr;
}

int mm_copy_header(const char* fname)
{
    FILE *f;
    FILE *out;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char outname[1000];
    sprintf(outname, "%s.%d", fname, rank);

    if ((f = fopen(fname, "r")) == NULL) return -1;
    if ((out = fopen(outname, "a")) == NULL) return -1;
   
    char line[1000];
    char first_char[10] = "%";
    while (1)
    {
        fgets(line, 1000, f);

        if (line[0] == first_char[0])
        {
            fprintf(out, "%s", line);
        }
        else
        {
            break;
        }
    }

    fclose(f);
    fclose(out);
}

int mm_write_lcl_size(const char* fname, index_t start, index_t stop)
{
    FILE *f;
    FILE *out;
    MM_typecode matcode;
    index_t M, N, nz;
    int i, ctr;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char outname[100];
    sprintf(outname, "%s.%d", fname, rank);

    if ((f = fopen(fname, "r")) == NULL) return -1;
    if ((out = fopen(outname, "a")) == NULL) return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("mm_read_unsymetric: Could not process Matrix Market banner ");
        printf(" in file [%s]\n", fname);
        return -1;
    }

    /* find out size of sparse matrix: M, N, nz .... */

    if (mm_read_mtx_crd_size(f, &M, &N, &nz) !=0)
    {
        fprintf(stderr, "read_unsymmetric_sparse(): could not parse matrix size.\n");
        return -1;
    }

    int lcl_nz = 0;
    int lcl_rows = stop - start;
    if (mm_is_integer(matcode) || mm_is_real(matcode))
    {
        for (i=0; i<nz; i++)
        {
            index_t row, col;
            data_t value;

            fscanf(f, "%d %d %lg\n", &row, &col, &value);

            if (row > start && row <= stop)
            {
                lcl_nz++;
            }
        }
        fprintf(out, "%d %d %d\n", M, N, nz);
        fprintf(out, "%d %d %d\n",  lcl_rows, lcl_rows, lcl_nz);
    }
   else
    {
        for (i=0; i<nz; i++)
        {
            index_t row, col;
            data_t value;

            fscanf(f, "%d %d\n", &row, &col);
            value = 1.0;

            if (row > start && row <= stop)
            {
                lcl_nz++;
            }
        }
        fprintf(out, "%d %d %d\n", M, N, nz);
        fprintf(out, "%d %d %d\n", lcl_rows, lcl_rows, lcl_nz);
    }

    fclose(f);
    fclose(out);

    return 0;

}

int mm_dist_sparse(const char* fname, index_t start, index_t stop)
{
    FILE *f;
    FILE *out;
    MM_typecode matcode;
    index_t M, N, nz;
    int i, ctr;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char outname[100];
    sprintf(outname, "%s.%d", fname, rank);

    if ((f = fopen(fname, "r")) == NULL)
        return -1;
    if ((out = fopen(outname, "a")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("mm_read_unsymetric: Could not process Matrix Market banner ");
        printf(" in file [%s]\n", fname);
        return -1;
    }

    /* find out size of sparse matrix: M, N, nz .... */

    if (mm_read_mtx_crd_size(f, &M, &N, &nz) !=0)
    {
        fprintf(stderr, "read_unsymmetric_sparse(): could not parse matrix size.\n");
        return -1;
    }

    int lcl_nz = 0;
    int lcl_rows = stop - start;
    if (mm_is_integer(matcode) || mm_is_real(matcode))
    {
        for (i=0; i<nz; i++)
        {
            index_t row, col;
            data_t value;

            fscanf(f, "%d %d %lg\n", &row, &col, &value);

            if (row > start && row <= stop)
            {
                fprintf(out, "%d %d %lg\n", row, col, value);
            }
        }
    }
    else
    {
        for (i=0; i<nz; i++)
        {
            index_t row, col;
            data_t value;

            fscanf(f, "%d %d\n", &row, &col);

            value = 1.0;

            if (row > start && row <= stop)
            {
                fprintf(out, "%d %d\n", row, col);
            }
        }
    }

    fclose(f);
    fclose(out);

    return 0;

}

int mm_read_sparse(const char *fname, index_t start, index_t stop, 
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
        fprintf(stderr, "Sorry, this application does not support ");
        fprintf(stderr, "Market Market type: [%s]\n",
                mm_typecode_to_str(matcode));
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


int mm_is_valid(MM_typecode matcode)
{
    if (!mm_is_matrix(matcode)) return 0;
    if (mm_is_dense(matcode) && mm_is_pattern(matcode)) return 0;
    if (mm_is_real(matcode) && mm_is_hermitian(matcode)) return 0;
    if (mm_is_pattern(matcode) && (mm_is_hermitian(matcode) || 
                mm_is_skew(matcode))) return 0;
    return 1;
}

int mm_read_banner(FILE *f, MM_typecode *matcode)
{
    char line[MM_MAX_LINE_LENGTH];
    char banner[MM_MAX_TOKEN_LENGTH];
    char mtx[MM_MAX_TOKEN_LENGTH]; 
    char crd[MM_MAX_TOKEN_LENGTH];
    char data_type[MM_MAX_TOKEN_LENGTH];
    char storage_scheme[MM_MAX_TOKEN_LENGTH];
    char *p;


    mm_clear_typecode(matcode);  

    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL) 
        return MM_PREMATURE_EOF;

    if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type, 
        storage_scheme) != 5)
        return MM_PREMATURE_EOF;

    for (p=mtx; *p!='\0'; *p=tolower(*p),p++);  /* convert to lower case */
    for (p=crd; *p!='\0'; *p=tolower(*p),p++);  
    for (p=data_type; *p!='\0'; *p=tolower(*p),p++);
    for (p=storage_scheme; *p!='\0'; *p=tolower(*p),p++);

    /* check for banner */
    if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
        return MM_NO_HEADER;

    /* first field should be "mtx" */
    if (strcmp(mtx, MM_MTX_STR) != 0)
        return  MM_UNSUPPORTED_TYPE;
    mm_set_matrix(matcode);


    /* second field describes whether this is a sparse matrix (in coordinate
            storgae) or a dense array */


    if (strcmp(crd, MM_SPARSE_STR) == 0)
        mm_set_sparse(matcode);
    else
    if (strcmp(crd, MM_DENSE_STR) == 0)
            mm_set_dense(matcode);
    else
        return MM_UNSUPPORTED_TYPE;
    

    /* third field */

    if (strcmp(data_type, MM_REAL_STR) == 0)
        mm_set_real(matcode);
    else
    if (strcmp(data_type, MM_COMPLEX_STR) == 0)
        mm_set_complex(matcode);
    else
    if (strcmp(data_type, MM_PATTERN_STR) == 0)
        mm_set_pattern(matcode);
    else
    if (strcmp(data_type, MM_INT_STR) == 0)
        mm_set_integer(matcode);
    else
        return MM_UNSUPPORTED_TYPE;
    

    /* fourth field */

    if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
        mm_set_general(matcode);
    else
    if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
        mm_set_symmetric(matcode);
    else
    if (strcmp(storage_scheme, MM_HERM_STR) == 0)
        mm_set_hermitian(matcode);
    else
    if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
        mm_set_skew(matcode);
    else
        return MM_UNSUPPORTED_TYPE;
        

    return 0;
}

int mm_write_mtx_crd_size(FILE *f, index_t M, index_t N, index_t nz)
{
    if (fprintf(f, "%d %d %d\n", M, N, nz) != 3)
        return MM_COULD_NOT_WRITE_FILE;
    else 
        return 0;
}

int mm_read_mtx_crd_size(FILE *f, index_t *M, index_t *N, index_t *nz )
{
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;

    /* set return null parameter values, in case we exit with errors */
    *M = *N = *nz = 0;

    /* now continue scanning until you reach the end-of-comments */
    do 
    {
        if (fgets(line,MM_MAX_LINE_LENGTH,f) == NULL) 
            return MM_PREMATURE_EOF;
    }while (line[0] == '%');

    /* line[] is either blank or has M,N, nz */
    if (sscanf(line, "%d %d %d", M, N, nz) == 3)
        return 0;
        
    else
    do
    { 
        num_items_read = fscanf(f, "%d %d %d", M, N, nz); 
        if (num_items_read == EOF) return MM_PREMATURE_EOF;
    }
    while (num_items_read != 3);

    return 0;
}


int mm_read_mtx_array_size(FILE *f, index_t *M, index_t *N)
{
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;
    /* set return null parameter values, in case we exit with errors */
    *M = *N = 0;
	
    /* now continue scanning until you reach the end-of-comments */
    do 
    {
        if (fgets(line,MM_MAX_LINE_LENGTH,f) == NULL) 
            return MM_PREMATURE_EOF;
    }while (line[0] == '%');

    /* line[] is either blank or has M,N, nz */
    if (sscanf(line, "%d %d", M, N) == 2)
        return 0;
        
    else /* we have a blank line */
    do
    { 
        num_items_read = fscanf(f, "%d %d", M, N); 
        if (num_items_read == EOF) return MM_PREMATURE_EOF;
    }
    while (num_items_read != 2);

    return 0;
}

int mm_write_mtx_array_size(FILE *f, index_t M, index_t N)
{
    if (fprintf(f, "%d %d\n", M, N) != 2)
        return MM_COULD_NOT_WRITE_FILE;
    else 
        return 0;
}

char *mm_strdup(const char *s)
{
	int len = strlen(s);
	char *s2 = (char *) malloc((len+1)*sizeof(char));
	return strcpy(s2, s);
}

char  *mm_typecode_to_str(MM_typecode matcode)
{
    char buffer[MM_MAX_LINE_LENGTH];
    char const *types[4];
	char *mm_strdup(const char *);

    /* check for MTX type */
    if (mm_is_matrix(matcode)) 
        types[0] = MM_MTX_STR;

    /* check for CRD or ARR matrix */
    if (mm_is_sparse(matcode))
        types[1] = MM_SPARSE_STR;
    else
    if (mm_is_dense(matcode))
        types[1] = MM_DENSE_STR;
    else
        return NULL;

    /* check for element data type */
    if (mm_is_real(matcode))
        types[2] = MM_REAL_STR;
    else
    if (mm_is_complex(matcode))
        types[2] = MM_COMPLEX_STR;
    else
    if (mm_is_pattern(matcode))
        types[2] = MM_PATTERN_STR;
    else
    if (mm_is_integer(matcode))
        types[2] = MM_INT_STR;
    else
        return NULL;


    /* check for symmetry type */
    if (mm_is_general(matcode))
        types[3] = MM_GENERAL_STR;
    else
    if (mm_is_symmetric(matcode))
        types[3] = MM_SYMM_STR;
    else 
    if (mm_is_hermitian(matcode))
        types[3] = MM_HERM_STR;
    else 
    if (mm_is_skew(matcode))
        types[3] = MM_SKEW_STR;
    else
        return NULL;

    sprintf(buffer,"%s %s %s %s", types[0], types[1], types[2], types[3]);
    return mm_strdup(buffer);

}
