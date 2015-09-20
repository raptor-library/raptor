#ifndef RAPTOR_UTILS_LINALG_SPMV_H
#define RAPTOR_UTILS_LINALG_SPMV_H

#include <mpi.h>
#include <float.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Core>
//using Eigen::VectorXd;

#include "core/vector.hpp"
#include "core/par_matrix.hpp"
#include "core/matrix.hpp"
#include "core/par_vector.hpp"
using namespace raptor;

//void sequentialSPMV(CSRMatrix* A, Vector* x, Vector* y, double alpha, double beta);
//void parallelSPMV(ParMatrix* A, ParVector* x, ParVector* y, double alpha, double beta);

void sequential_spmv(Matrix* A, Vector* x, Vector* y, double alpha, double beta);
void sequential_spmv_T(Matrix* A, Vector* x, Vector* y, double alpha, double beta);
void sequential_spmv(Matrix* A, Vector* x, Vector* y, double alpha, double beta, std::vector<index_t> col_list);
void sequential_spmv_T(Matrix* A, Vector* x, Vector* y, double alpha, double beta, std::vector<index_t> col_list);

void parallel_spmv(ParMatrix* A, ParVector* x, ParVector* y, data_t alpha, data_t beta, index_t async = 0)
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

    if (recv_procs.size())
    {
        recv_requests = new MPI_Request [recv_procs.size()];
        for (index_t i = 0; i < recv_procs.size(); i++)
        {
            recv_requests[i] = MPI_REQUEST_NULL;
        }
        recv_buffer = new data_t [comm->size_recvs];

        // Send and receive vector data
        // Begin sending and gathering off-diagonal entries
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

    if (send_procs.size())
    {
        // TODO we do not want to malloc these every time
        send_requests = new MPI_Request [send_procs.size()];
        for (index_t i = 0; i < send_procs.size(); i++)
        {
            send_requests[i] = MPI_REQUEST_NULL;
        }
        send_buffer = new data_t [comm->size_sends];

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
    sequential_spmv(A->diag, x->local, y->local, alpha, beta);

    // Once data is available, add contribution of off-diagonals
    // TODO Deal with new entries as they become available
    // TODO Add an error check on the status
    // TODO Using MPI_STATUS_IGNORE (delete[] recvStatus was causing a segfault)
    if (recv_procs.size())
    {
        if (async)
        {
            for (index_t i = 0; i < recv_procs.size(); i++)
            {
                index_t recv_idx = 0;
                MPI_Waitany(recv_procs.size(), recv_requests, &recv_idx, MPI_STATUS_IGNORE);
                index_t proc = recv_procs[recv_idx];
                Vector offd_tmp = Eigen::Map<Vector>(&recv_buffer[recv_proc_starts[proc]], recv_indices[proc].size());
                sequential_spmv(A->offd, &offd_tmp, y->local, alpha, 1.0, recv_indices[proc]);
            }
        }
        else
        {
            // Wait for all receives to finish
            MPI_Waitall(recv_procs.size(), recv_requests, MPI_STATUS_IGNORE);

            // Add received data to Vector
            Vector offd_tmp = Eigen::Map<Vector>(recv_buffer, tmp_size);
            sequential_spmv(A->offd, &offd_tmp, y->local, alpha, 1.0); 

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

//alpha*A^Tx + beta*y 
void parallel_spmv_T(ParMatrix* A, ParVector* x, ParVector* y, data_t alpha, data_t beta, index_t async = 0)
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
                sequential_spmv_T(A->offd, x->local, offd_tmp, alpha, 0.0, send_indices[proc]);
                local_data = offd_tmp->data();

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
            sequential_spmv_T(A->offd, x->local, offd_tmp, alpha, 0.0);
            local_data = offd_tmp->data();

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

    sequential_spmv_T(A->diag, x->local, y->local, alpha, beta);

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

#endif
