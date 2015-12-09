#include "jacobi.hpp"
#include "spmv.hpp"

void jacobi_diag(const Matrix* A, const data_t* x, data_t* result, data_t* diag_data)
{
    const std::vector<index_t>& indptr = A->indptr;
    const std::vector<index_t>& indices = A->indices;
    const std::vector<data_t>& data = A->data;

    index_t num_rows = A->n_rows;
    index_t row_start, row_end;

    index_t col;
    data_t value;

    for (index_t row = 0; row < num_rows; row++)
    {
        row_start = indptr[row];
        row_end = indptr[row+1];

        for (index_t j = row_start; j < row_end; j++)
        {
            col = indices[j];
            value = data[j];
            if (col == row)
            {
                diag_data[row] = value;
            }
            else
            {
                result[row] += value * x[col]; 
            }
        }
    }
}

void jacobi(const ParMatrix* A, ParVector* x, const ParVector* y, const data_t omega, const int async)
{
    if (A->local_rows == 0) return;

    // Get MPI Information
    MPI_Comm comm_mat = A->comm_mat;
    int rank, num_procs;
    MPI_Comm_rank(comm_mat, &rank);
    MPI_Comm_size(comm_mat, &num_procs);

    // TODO must enforce that y and x are not aliased, or this will NOT work
    //    or we could use blocking sends as long as we post the iRecvs first

    // Declare communication variables
    MPI_Request*                             send_requests;
    MPI_Request*                             recv_requests;
    data_t*                                  send_buffer;
    data_t*                                  recv_buffer;
    data_t*                                  local_data;
    Vector                                   offd_tmp;
    index_t                                  begin;
    index_t                                  ctr;
    index_t                                  request_ctr;
    index_t                                  send_size;
    index_t                                  recv_size;
    index_t                                  num_sends;
    index_t                                  num_recvs;
    index_t*                                 send_procs;
    index_t*                                 recv_procs;
    ParComm*                                 comm;
    std::map<index_t, index_t>               recv_proc_starts;

    // Initialize communication variables
    comm          = A->comm;
    send_procs    = comm->send_procs.data();
    recv_procs    = comm->recv_procs.data();
    local_data    = x->local->data();
    num_sends = comm->send_procs.size();
    num_recvs = comm->recv_procs.size();
    send_size = comm->size_sends;
    recv_size = comm->size_recvs;

    data_t* x_data = x->local->data();
    data_t* y_data = y->local->data();
    data_t* sum_data = new data_t[y->local_n]();
    data_t* diag_data = new data_t[A->diag->n_rows];

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
        for (index_t i = 0; i < num_recvs; i++)
        {
            index_t proc = recv_procs[i];
            index_t num_recv = comm->recv_indices[proc].size();
            recv_proc_starts[proc] = begin;
            MPI_Irecv(&recv_buffer[begin], num_recv, MPI_DATA_T, proc, 0, comm_mat, &(recv_requests[request_ctr++]));
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
        for (index_t i = 0; i < num_sends; i++)
        {
            index_t proc = send_procs[i];
            ctr = 0;
            std::vector<index_t>* s_indices = &(comm->send_indices[proc]);
            for (auto send_idx : *s_indices)
            {
                send_buffer[begin + ctr] = local_data[send_idx];
                ctr++;
            }
            MPI_Isend(&send_buffer[begin], ctr, MPI_DATA_T, proc, 0, comm_mat, &(send_requests[request_ctr++]));
            begin += ctr;
        }
    }

    // Compute partial SpMV with local information
    jacobi_diag(A->diag, x_data, sum_data, diag_data);

    // Once data is available, add contribution of off-diagonals
    // TODO Deal with new entries as they become available
    // TODO Add an error check on the status
    if (num_recvs)
    {
        if (async)
        {
            int recv_idx[num_recvs];
            for (index_t i = 0; i < num_recvs;)
            {
                int n_recvd;
                MPI_Waitsome(num_recvs, recv_requests, &n_recvd, recv_idx, MPI_STATUS_IGNORE);
                for (index_t j = 0; j < n_recvd; j++)
                {
                    index_t proc = recv_procs[recv_idx[j]];
                    std::vector<index_t>* r_indices = &(comm->recv_indices[proc]);
                    int first_col = r_indices->data()[0];
                    int num_cols = r_indices->size();
                    sequential_spmv(A->offd, x_data, sum_data, 1.0, 1.0, NULL, first_col, num_cols);
                }
                i += n_recvd;
            }
        }
        else
        {
            // Wait for all receives to finish
            MPI_Waitall(num_recvs, recv_requests, MPI_STATUS_IGNORE);

            // Add received data to Vector
            sequential_spmv(A->offd, recv_buffer, sum_data, 1.0, 1.0); 

        }

    	delete[] recv_requests; 
        delete[] recv_buffer;
    }

    index_t num_rows = A->local_rows;
    for (int i = 0; i < num_rows; i++)
    {
        if (fabs(diag_data[i]) > zero_tol)
        {
            x_data[i] = (omega * (y_data[i] - sum_data[i]) / diag_data[i]) + ((1 - omega) * x_data[i]);
        }
    }

    delete[] diag_data;
    delete[] sum_data;

    if (num_sends)
    {
        // Wait for all sends to finish
        // TODO Add an error check on the status
        MPI_Waitall(num_sends, send_requests, MPI_STATUS_IGNORE);

        // Delete MPI_Requests
        delete[] send_requests; 
        delete[] send_buffer;
    }
}


