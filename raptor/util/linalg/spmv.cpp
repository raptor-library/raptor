#include "spmv.hpp"

/**************************************************************
 *****   Sequential Matrix-Vector Multiplication
 **************************************************************
 ***** Performs matrix-vector multiplication on inner indices
 ***** y[inner] = alpha * A[inner, outer] * x[outer] + beta*y[inner]
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled
 ***** x : Vector*
 *****    Vector to be multiplied
 ***** y : Vector*
 *****    Vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy A*x by
 ***** beta : data_t
 *****    Scalar to multiply original y by
 **************************************************************/
void seq_inner_spmv(Matrix* A, data_t* x, data_t* y, data_t alpha, data_t beta)
{
    index_t alpha_zero = (fabs(alpha) < zero_tol);
    index_t alpha_one = (fabs(alpha - 1.0) < zero_tol);
    index_t alpha_neg_one = (fabs(alpha + 1.0) < zero_tol);

    index_t beta_zero = (fabs(beta) < zero_tol);
    index_t beta_one = (fabs(beta - 1.0) < zero_tol);
    index_t beta_neg_one = (fabs(beta + 1.0) < zero_tol);

    std::vector<index_t> ptr = A->indptr;
    std::vector<index_t> idx = A->indices;
    std::vector<data_t> values = A->data;
    index_t num_cols = A->n_cols;
    index_t num_rows = A->n_rows;
    index_t n_outer = A->n_outer;
    index_t n_inner = A->n_inner;

    index_t ptr_start;
    index_t ptr_end;
    data_t x_val;

    if (alpha_one)
    {
        if (!beta_one)
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                y[inner] = beta * y[inner];
            }
        }
        for (index_t outer = 0; outer < n_outer; outer++)
        {
            ptr_start = ptr[outer];
            ptr_end = ptr[outer + 1];
            data_t x_val = x[outer];
            for (index_t j = ptr_start; j < ptr_end; j++)
            {
                index_t inner = idx[j];
                y[inner] += values[j] * x_val;
            }
        }
    }
    else if (alpha_neg_one)
    {
        if (!beta_one)
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                y[inner] = beta * y[inner];
            }
        }
        for (index_t outer = 0; outer < n_outer; outer++)
        {
            ptr_start = ptr[outer];
            ptr_end = ptr[outer + 1];
            data_t x_val = x[outer];
            for (index_t j = ptr_start; j < ptr_end; j++)
            {
                index_t inner = idx[j];
                y[inner] -= values[j] * x_val;
            }
        }
    }
    else if (alpha_zero)
    {
        if (beta_zero)
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                y[inner] = 0.0;
            }
        }
        else if (!beta_one)
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                y[inner] = beta * y[inner];
            }
        }
    }
    else
    {
        if (!beta_one)
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                y[inner] = beta * y[inner];
            }
        }
        for (index_t outer = 0; outer < n_outer; outer++)
        {
            ptr_start = ptr[outer];
            ptr_end = ptr[outer + 1];
            data_t x_val = x[outer];
            for (index_t j = ptr_start; j < ptr_end; j++)
            {
                index_t inner = idx[j];
                y[inner] += alpha * values[j] * x_val;
            }
        }
    }
}

/**************************************************************
 *****   Partial Sequential Matrix-Vector Multiplication
 **************************************************************
 ***** Performs partial matrix-vector multiplication on inner indices
 ***** y[inner] = alpha * A[inner, outer] * x[outer] + beta*y[inner]
 ***** for a portion of the outer indices.
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled
 ***** x : Vector*
 *****    Vector to be multiplied
 ***** y : Vector*
 *****    Vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy A*x by
 ***** beta : data_t
 *****    Scalar to multiply original y by
 ***** outer_list : std::vector<index_t>
 *****    Outer indices to multiply
 **************************************************************/
void seq_inner_spmv(Matrix* A, data_t* x, data_t* y, data_t alpha, data_t beta, std::vector<index_t> outer_list)
{
    index_t alpha_zero = (fabs(alpha) < zero_tol);
    index_t alpha_one = (fabs(alpha - 1.0) < zero_tol);
    index_t alpha_neg_one = (fabs(alpha + 1.0) < zero_tol);

    index_t beta_zero = (fabs(beta) < zero_tol);
    index_t beta_one = (fabs(beta - 1.0) < zero_tol);
    index_t beta_neg_one = (fabs(beta + 1.0) < zero_tol);

    std::vector<index_t> ptr = A->indptr;
    std::vector<index_t> idx = A->indices;
    std::vector<data_t> values = A->data;
    index_t num_cols = A->n_cols;
    index_t num_rows = A->n_rows;
    index_t n_outer = A->n_outer;
    index_t n_inner = A->n_inner;
    index_t num_nonzeros = A->nnz;

    index_t ptr_start;
    index_t ptr_end;
    data_t x_val;

    if (alpha_one)
    {
        if (!beta_one)
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                y[inner] = beta * y[inner];
            }
        }
        // Ax + y
        for (index_t i = 0; i < outer_list.size(); i++)
        {
            index_t outer = outer_list[i];
            ptr_start = ptr[outer];
            ptr_end = ptr[outer + 1];
            data_t x_val = x[i];

            for (index_t j = ptr_start; j < ptr_end; j++)
            {
                index_t inner = idx[j];
                y[inner] += values[j] * x_val;
            }
        }
    }
    else if (alpha_neg_one)
    {
        if (!beta_one)
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                y[inner] = beta * y[inner];
            }
        }
        for (index_t i = 0; i < outer_list.size(); i++)
        {
            index_t outer = outer_list[i];
            ptr_start = ptr[outer];
            ptr_end = ptr[outer + 1];
            data_t x_val = x[i];
            for (index_t j = ptr_start; j < ptr_end; j++)
            {
                index_t inner = idx[j];
                y[inner] -= values[j] * x_val;
            }
        }
    }
    else if (alpha_zero)
    {
        if (beta_zero)
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                y[inner] = 0.0;
            }
        }
        else if (!beta_one)
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                y[inner] = beta * y[inner];
            }
        }
    }
    else
    {
        if (!beta_one)
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                y[inner] = beta * y[inner];
            }
        }
        for (index_t i = 0; i < outer_list.size(); i++)
        {
            index_t outer = outer_list[i];
            ptr_start = ptr[outer];
            ptr_end = ptr[outer + 1];
            data_t x_val = x[i];
            for (index_t j = ptr_start; j < ptr_end; j++)
            {
                index_t inner = idx[j];
                y[inner] += alpha * values[j] * x_val;
            }
        }
    }
}

/**************************************************************
 *****   Sequential Matrix-Vector Multiplication
 **************************************************************
 ***** Performs matrix-vector multiplication on outer indices
 ***** y[outer] = alpha * A[outer, inner] * x[inner] + beta*y[outer]
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled
 ***** x : Vector*
 *****    Vector to be multiplied
 ***** y : Vector*
 *****    Vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy A*x by
 ***** beta : data_t
 *****    Scalar to multiply original y by
 **************************************************************/
void seq_outer_spmv(Matrix* A, data_t* x, data_t* y, data_t alpha, data_t beta)
{
    index_t alpha_zero = (fabs(alpha) < zero_tol);
    index_t alpha_one = (fabs(alpha - 1.0) < zero_tol);
    index_t alpha_neg_one = (fabs(alpha + 1.0) < zero_tol);

    index_t beta_zero = (fabs(beta) < zero_tol);
    index_t beta_one = (fabs(beta - 1.0) < zero_tol);
    index_t beta_neg_one = (fabs(beta + 1.0) < zero_tol);

    std::vector<index_t> ptr = A->indptr;
    std::vector<index_t> idx = A->indices;
    std::vector<data_t> values = A->data;
    index_t num_cols = A->n_cols;
    index_t num_rows = A->n_rows;
    index_t n_outer = A->n_outer;
    index_t n_inner = A->n_inner;
    index_t num_nonzeros = A->nnz;

    index_t ptr_start;
    index_t ptr_end;

    if (alpha_one)
    {
        if (beta_one)
        {
            //Ax + y
            for (index_t outer = 0; outer < n_outer; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                for (index_t j = ptr_start; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    y[outer] += values[j] * x[inner];
                }
            }
        }
        else if (beta_zero)
        {
            //Ax
            for (index_t outer = 0; outer < n_outer; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                if (ptr_start < ptr_end)
                {
                    y[outer] = values[ptr_start] * x[idx[ptr_start]];
                }
                else
                {
                    y[outer] = 0.0;
                }

                for (index_t j = ptr_start + 1; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    y[outer] += values[j] * x[inner];
                }
            }
        }
        else
        {
            //Ax + beta * y
            for (index_t outer = 0; outer < n_outer; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                y[outer] *= beta;

                for (index_t j = ptr_start + 1; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    y[outer] += values[j] * x[inner];
                }
            }
        }
    }
    else if (alpha_neg_one)
    {
        if (beta_one)
        {
            //-Ax + y
            for (index_t outer = 0; outer < n_outer; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                for (index_t j = ptr_start; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    y[outer] -= values[j] * x[inner];
                }
            }
        }
        else if (beta_zero)
        {
            //Ax
            for (index_t outer = 0; outer < n_outer; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                if (ptr_start < ptr_end)
                {
                    y[outer] = - (values[ptr_start] * x[idx[ptr_start]]);
                }
                else
                {
                    y[outer] = 0.0;
                }

                for (index_t j = ptr_start + 1; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    y[outer] -= values[j] * x[inner];
                }
            }
        }
        else
        {
            //Ax + beta * y
            for (index_t outer = 0; outer < n_outer; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                y[outer] *= beta;

                for (index_t j = ptr_start + 1; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    y[outer] -= values[j] * x[inner];
                }
            }
        }
    }
    else if (alpha_zero)
    {
        if (beta_zero)
        {
            //return 0
            for (index_t outer = 0; outer < n_outer; outer++)
            {
                y[outer] = 0.0;
            }
        }
        else if (!beta_one)
        {
            //beta * y
            for (index_t outer = 0; outer < n_outer; outer++)
            {
                y[outer] *= beta;
            }
        }
    }
    else
    {
        if (beta_one)
        {
            //alpha*Ax + y
            for (index_t outer = 0; outer < n_outer; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                for (index_t j = ptr_start; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    y[outer] += alpha * values[j] * x[inner];
                }
            }
        }
        else if (beta_zero)
        {
            //alpha*Ax
            for (index_t outer = 0; outer < n_outer; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                if (ptr_start < ptr_end)
                {
                    y[outer] = alpha * values[ptr_start] * x[idx[ptr_start]];
                }
                else
                {
                    y[outer] = 0.0;
                }

                for (index_t j = ptr_start + 1; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    y[outer] += alpha * values[j] * x[inner];
                }
            }
        }
        else
        {
            //alpha * Ax + beta * y
            for (index_t outer = 0; outer < n_outer; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                y[outer] *= beta;

                for (index_t j = ptr_start + 1; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    y[outer] += alpha * values[j] * x[inner];
                }
            }
        }
    }
}

/**************************************************************
 *****   Partial Sequential Matrix-Vector Multiplication
 **************************************************************
 ***** Performs partial matrix-vector multiplication on inner indices
 ***** y[outer] = alpha * A[outer, inner] * x[inner] + beta*y[outer]
 ***** for a portion of the outer indices.
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled
 ***** x : Vector*
 *****    Vector to be multiplied
 ***** y : Vector*
 *****    Vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy A*x by
 ***** beta : data_t
 *****    Scalar to multiply original y by
 ***** outer_list : std::vector<index_t>
 *****    Outer indices to multiply
 **************************************************************/
void seq_outer_spmv(Matrix* A, data_t* x, data_t* y, data_t alpha, data_t beta, std::vector<index_t> outer_list)
{
    index_t alpha_zero = (fabs(alpha) < zero_tol);
    index_t alpha_one = (fabs(alpha - 1.0) < zero_tol);
    index_t alpha_neg_one = (fabs(alpha + 1.0) < zero_tol);

    index_t beta_zero = (fabs(beta) < zero_tol);
    index_t beta_one = (fabs(beta - 1.0) < zero_tol);
    index_t beta_neg_one = (fabs(beta + 1.0) < zero_tol);

    std::vector<index_t> ptr = A->indptr;
    std::vector<index_t> idx = A->indices;
    std::vector<data_t> values = A->data;
    index_t num_cols = A->n_cols;
    index_t num_rows = A->n_rows;
    index_t n_outer = A->n_outer;
    index_t n_inner = A->n_inner;
    index_t num_nonzeros = A->nnz;

    index_t ptr_start;
    index_t ptr_end;

    if (alpha_one)
    {
        if (beta_one)
        {
            //Ax + y
            for (index_t i = 0; i < outer_list.size(); i++)
            {
                index_t outer = outer_list[i];
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                for (index_t j = ptr_start; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    y[i] += values[j] * x[inner];
                }
            }
        }
        else
        {
            //Ax + beta*y
            for (index_t i = 0; i < outer_list.size(); i++)
            {
                index_t outer = outer_list[i];
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                y[i] *= beta;

                for (index_t j = ptr_start; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    y[i] += values[j] * x[inner];
                }
            }
        }
    }
    else if (alpha_neg_one)
    {
        if (beta_one)
        {
            //Ax + y
            for (index_t i = 0; i < outer_list.size(); i++)
            {
                index_t outer = outer_list[i];
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                for (index_t j = ptr_start; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    y[i] -= values[j] * x[inner];
                }
            }
        }
        else
        {
            //Ax + beta*y
            for (index_t i = 0; i < outer_list.size(); i++)
            {
                index_t outer = outer_list[i];
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                y[i] *= beta;

                for (index_t j = ptr_start; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    y[i] -= values[j] * x[inner];
                }
            }
        }
    }
    else if (alpha_zero)
    {
        if (beta_zero)
        {
            //return 0
            for (index_t i = 0; i < outer_list.size(); i++)
            {
                y[i] = 0.0;
            }
        }
        else if (!beta_one)
        {
            //beta * y
            for (index_t i = 0; i < outer_list.size(); i++)
            {
                y[i] *= beta;
            }
        }
    }
    else
    {
        if (beta_one)
        {
            //Ax + y
            for (index_t i = 0; i < outer_list.size(); i++)
            {
                index_t outer = outer_list[i];
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                for (index_t j = ptr_start; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    y[i] += alpha * values[j] * x[inner];
                }
            }
        }
        else
        {
            //Ax + beta*y
            for (index_t i = 0; i < outer_list.size(); i++)
            {
                index_t outer = outer_list[i];
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                y[i] *= beta;

                for (index_t j = ptr_start; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    y[i] += alpha * values[j] * x[inner];
                }
            }
        }
    }
}

/**************************************************************
 *****   Sequential Matrix-Vector Multiplication
 **************************************************************
 ***** Performs partial matrix-vector multiplication, calling
 ***** method appropriate for matrix format
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled
 ***** x : Vector*
 *****    Vector to be multiplied
 ***** y : Vector*
 *****    Vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy A*x by
 ***** beta : data_t
 *****    Scalar to multiply original y by
 **************************************************************/
void sequential_spmv(Matrix* A, data_t* x, data_t* y, data_t alpha, data_t beta)
{
    if (A->format == CSR)
    {
        seq_outer_spmv(A, x, y, alpha, beta);
    }
    else
    {
        seq_inner_spmv(A, x, y, alpha, beta);
    }   
}

/**************************************************************
 *****   Partial Sequential Matrix-Vector Multiplication
 **************************************************************
 ***** Performs partial matrix-vector multiplication, calling
 ***** method appropriate for matrix format
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled
 ***** x : Vector*
 *****    Vector to be multiplied
 ***** y : Vector*
 *****    Vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy A*x by
 ***** beta : data_t
 *****    Scalar to multiply original y by
 ***** outer_list : std::vector<index_t>
 *****    Outer indices to multiply
 **************************************************************/
void sequential_spmv(Matrix* A, data_t* x, data_t* y, data_t alpha, data_t beta, std::vector<index_t> col_list)
{
    if (A->format == CSR)
    {
        
    }
    else
    {
        seq_inner_spmv(A, x, y, alpha, beta, col_list);
    }   
}

/**************************************************************
 *****   Sequential Transpose Matrix-Vector Multiplication
 **************************************************************
 ***** Performs partial transpose matrix-vector multiplication, 
 ***** calling method appropriate for matrix format
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled
 ***** x : Vector*
 *****    Vector to be multiplied
 ***** y : Vector*
 *****    Vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy A*x by
 ***** beta : data_t
 *****    Scalar to multiply original y by
 **************************************************************/
void sequential_spmv_T(Matrix* A, data_t* x, data_t* y, data_t alpha, data_t beta)
{
    if (A->format == CSR)
    {
        seq_inner_spmv(A, x, y, alpha, beta);
    }
    else
    {
        seq_outer_spmv(A, x, y, alpha, beta);
    }   
}

/**************************************************************
 *****   Partial Sequential Transpose Matrix-Vector Multiplication
 **************************************************************
 ***** Performs partial transpose matrix-vector multiplication,
 ***** calling method appropriate for matrix format
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled
 ***** x : Vector*
 *****    Vector to be multiplied
 ***** y : Vector*
 *****    Vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy A*x by
 ***** beta : data_t
 *****    Scalar to multiply original y by
 ***** outer_list : std::vector<index_t>
 *****    Outer indices to multiply
 **************************************************************/
void sequential_spmv_T(Matrix* A, data_t* x, data_t* y, data_t alpha, data_t beta, std::vector<index_t> col_list)
{
    if (A->format == CSR)
    {

    }
    else
    {
        seq_outer_spmv(A, x, y, alpha, beta, col_list);
    }   
}

/**************************************************************
 *****   Parallel Matrix-Vector Multiplication
 **************************************************************
 ***** Performs parallel matrix-vector multiplication
 ***** y = alpha*A*x + beta*y
 *****
 ***** Parameters
 ***** -------------
 ***** A : ParMatrix*
 *****    Parallel matrix to be multipled
 ***** x : ParVector*
 *****    Parallel vector to be multiplied
 ***** y : ParVector*
 *****    Parallel vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy alpha*A*x
 ***** beta : data_t
 *****    Scalar to multiply original y by
 ***** async : index_t
 *****    Boolean flag for updating SpMV asynchronously
 **************************************************************/
void parallel_spmv(ParMatrix* A, ParVector* x, ParVector* y, data_t alpha, data_t beta, index_t async)
{
    // Get MPI Information
    index_t rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // TODO must enforce that y and x are not aliased, or this will NOT work
    //    or we could use blocking sends as long as we post the iRecvs first

    // Declare communication variables
    MPI_Request*                            send_requests;
    MPI_Request*                            recv_requests;
    data_t*                                 send_buffer;
    data_t*                                 recv_buffer;
    std::map<index_t, index_t>              recv_proc_starts;
    data_t*                                 local_data;
    Vector                                  offd_tmp;
    index_t                                 tmp_size;
    index_t                                 begin;
    index_t                                 ctr;
    index_t                                 request_ctr;
    index_t                                 tag;
    index_t                                 send_size;
    index_t                                 recv_size;
    index_t                                 num_sends;
    index_t                                 num_recvs;
    ParComm*                                comm;
    std::vector<index_t>                    send_procs;
    std::vector<index_t>                    recv_procs;
    std::map<index_t, std::vector<index_t>> send_indices;
    std::map<index_t, std::vector<index_t>> recv_indices;

    // Initialize communication variables
    comm          = A->comm;
    send_procs    = comm->send_procs;
    recv_procs    = comm->recv_procs;
    send_indices  = comm->send_indices;
    recv_indices  = comm->recv_indices;
    tmp_size      = comm->size_recvs;
    local_data    = x->local->data();
    num_sends = send_procs.size();
    num_recvs = recv_procs.size();
    send_size = comm->size_sends;
    recv_size = comm->size_recvs;

    // If receive values, post appropriate MPI Receives
    if (num_recvs)
    {
        // Initialize recv requests and buffer
        recv_requests = new MPI_Request [num_recvs];
        for (index_t i = 0; i < num_recvs; i++)
        {
            recv_requests[i] = MPI_REQUEST_NULL;
        }
        recv_buffer = new data_t [recv_size];

        // Post receives for x-values that are needed
        begin = 0;
        ctr = 0;
        request_ctr = 0;
        for (auto proc : recv_procs)
        {
            index_t num_recv = recv_indices[proc].size();
            recv_proc_starts[proc] = begin;
            MPI_Irecv(&recv_buffer[begin], num_recv, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, &(recv_requests[request_ctr++]));
            begin += num_recv;
        }
    
    }

    // Send values of x to appropriate processors
    if (num_sends)
    {
        // TODO we do not want to malloc these every time
        send_requests = new MPI_Request [num_sends];
        for (index_t i = 0; i < num_sends; i++)
        {
            send_requests[i] = MPI_REQUEST_NULL;
        }
        send_buffer = new data_t [send_size];

        begin = 0;
        request_ctr = 0;
        for (auto proc : send_procs)
        {
            ctr = 0;
            for (auto send_idx : send_indices[proc])
            {
                send_buffer[begin + ctr] = local_data[send_idx];
                ctr++;
            }
            MPI_Isend(&send_buffer[begin], ctr, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, &(send_requests[request_ctr++]));
            begin += ctr;
        }
    }

    // Compute partial SpMV with local information
    sequential_spmv(A->diag, x->local->data(), y->local->data(), alpha, beta);

    // Once data is available, add contribution of off-diagonals
    // TODO Deal with new entries as they become available
    // TODO Add an error check on the status
    if (num_recvs)
    {
        if (async)
        {
            for (index_t i = 0; i < recv_procs.size(); i++)
            {
                index_t n_recvd;
                index_t recv_idx[num_recvs];
                MPI_Waitsome(num_recvs, recv_requests, &n_recvd, recv_idx, MPI_STATUS_IGNORE);
                for (index_t j = 0; j < n_recvd; j++)
                {
                    index_t proc = recv_procs[recv_idx[j]];
                    sequential_spmv(A->offd, &recv_buffer[recv_proc_starts[proc]], y->local->data(), alpha, 1.0, recv_indices[proc]);
                }
            }
        }
        else
        {
            // Wait for all receives to finish
            MPI_Waitall(recv_procs.size(), recv_requests, MPI_STATUS_IGNORE);

            // Add received data to Vector
            sequential_spmv(A->offd, recv_buffer, y->local->data(), alpha, 1.0); 

        }

    	delete[] recv_requests; 
        delete[] recv_buffer;
    }

    if (num_sends)
    {
        // Wait for all sends to finish
        // TODO Add an error check on the status
        MPI_Waitall(send_procs.size(), send_requests, MPI_STATUS_IGNORE);

        // Delete MPI_Requests
        delete[] send_requests; 
        delete[] send_buffer;
    }
}

/**************************************************************
 *****   Parallel Transpose Matrix-Vector Multiplication
 **************************************************************
 ***** Performs parallel transpose matrix-vector multiplication
 ***** y = alpha*A^T*x + beta*y
 *****
 ***** Parameters
 ***** -------------
 ***** A : ParMatrix*
 *****    Parallel matrix to be transposed and multipled
 ***** x : ParVector*
 *****    Parallel vector to be multiplied
 ***** y : ParVector*
 *****    Parallel vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy alpha*A*x
 ***** beta : data_t
 *****    Scalar to multiply original y by
 ***** async : index_t
 *****    Boolean flag for updating SpMV asynchronously
 **************************************************************/
void parallel_spmv_T(ParMatrix* A, ParVector* x, ParVector* y, data_t alpha, data_t beta, index_t async)
{
    // Get MPI Information
    index_t rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Declare communication variables
	MPI_Request*                            send_requests;
    MPI_Request*                            recv_requests;
    data_t*                                 send_buffer;
    data_t*                                 recv_buffer;
    std::map<index_t, index_t>              recv_proc_starts;
    data_t*                                 local_data;
    Vector*                                  offd_tmp;
    index_t                                 tmp_size;
    index_t                                 begin;
    index_t                                 ctr;
    index_t                                 request_ctr;
    index_t                                 msg_tag;
    index_t                                 size_sends;
    index_t                                 size_recvs;
    ParComm*                                comm;
    std::vector<index_t>                    send_procs;
    std::vector<index_t>                    recv_procs;
    std::map<index_t, std::vector<index_t>> send_indices;
    std::map<index_t, std::vector<index_t>> recv_indices;

    // Initialize communication variables
    comm          = A->comm;
    send_procs    = comm->recv_procs;
    recv_procs    = comm->send_procs;
    send_indices  = comm->recv_indices;
    recv_indices  = comm->send_indices;
    size_sends    = comm->size_recvs;
    size_recvs    = comm->size_sends;
    tmp_size      = size_sends;
    msg_tag       = 1111;


    if (recv_procs.size())
    {
        recv_requests = new MPI_Request [recv_procs.size()];
        recv_buffer = new data_t [size_recvs];

        // Send and receive vector data
	    // Begin sending and gathering off-diagonal entries
        begin = 0;
        request_ctr = 0;
        for (auto proc : recv_procs)
        {
            index_t num_recv = recv_indices[proc].size();
            recv_proc_starts[proc] = begin;
            MPI_Irecv(&recv_buffer[begin], num_recv, MPI_DOUBLE, proc, msg_tag, MPI_COMM_WORLD, &(recv_requests[request_ctr++]));
            begin += num_recv;
        }   
    }


    if (send_procs.size())
    {
        // TODO we do not want to malloc these every time
        send_requests = new MPI_Request [send_procs.size()];
        send_buffer = new data_t [size_sends];
        offd_tmp = new Vector(tmp_size);

        begin = 0;
        request_ctr = 0;
        ctr = 0;

        if (async)
        {
            for (auto proc : send_procs)
            {
                std::vector<index_t> tmp_indices = send_indices[proc];
                offd_tmp->resize(send_indices[proc].size());
                local_data = offd_tmp->data();
                sequential_spmv_T(A->offd, x->local->data(), local_data, alpha, 0.0, send_indices[proc]);

                ctr = 0;
                for (index_t i = 0; i < tmp_indices.size(); i++)
                {
                    send_buffer[begin + ctr] = local_data[i];
                    ctr++;
                }
                MPI_Isend(&send_buffer[begin], ctr, MPI_DOUBLE, proc, msg_tag, MPI_COMM_WORLD, &(send_requests[request_ctr++]));
                begin += ctr;
            }
        }
        else
        {
            local_data = offd_tmp->data();
            sequential_spmv_T(A->offd, x->local->data(), local_data, alpha, 0.0);

	        for (auto proc : send_procs)
            {
                ctr = 0;
                for (auto send_idx : send_indices[proc])
                {
                    send_buffer[begin + ctr] = local_data[send_idx];
                    ctr++;
                }
                MPI_Isend(&send_buffer[begin], ctr, MPI_DOUBLE, proc, msg_tag, MPI_COMM_WORLD, &(send_requests[request_ctr++]));
                begin += ctr;
            }
        }
        delete offd_tmp;
    }

    sequential_spmv_T(A->diag, x->local->data(), y->local->data(), alpha, beta);

    if (recv_procs.size())
    {
        local_data = y->local->data();
        if (async)
        {
            for (index_t i = 0; i < recv_procs.size(); i++)
            {
                index_t recv_idx = 0;
                MPI_Waitany(recv_procs.size(), recv_requests, &recv_idx, MPI_STATUS_IGNORE);
                index_t proc = recv_procs[recv_idx];
                std::vector<index_t> tmp_indices = recv_indices[proc];
                for (index_t j = 0; j < tmp_indices.size(); j++)
                {
                    index_t row = tmp_indices[j];
                    local_data[row] += recv_buffer[recv_proc_starts[proc] + j];
                }
            }
        }
        else
        {
            // Wait for all receives to finish
            MPI_Waitall(recv_procs.size(), recv_requests, MPI_STATUS_IGNORE);

            for (auto proc : recv_procs)
            {
                std::vector<index_t> tmp_indices = recv_indices[proc];
                for (index_t j = 0; j < tmp_indices.size(); j++)
                {
                    index_t row = tmp_indices[j];
                    local_data[row] += recv_buffer[recv_proc_starts[proc] + j];
                }
            }
        }

    	delete[] recv_requests; 
        delete[] recv_buffer;
    } 

    if (send_procs.size())
    {
	    // Wait for all sends to finish
	    // TODO Add an error check on the status
	    MPI_Waitall(send_procs.size(), send_requests, MPI_STATUS_IGNORE);

        // Delete MPI_Requests
        delete[] send_requests; 
        delete[] send_buffer;
    } 
}
