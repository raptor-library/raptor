// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_PARCOMM_HPP
#define RAPTOR_CORE_PARCOMM_HPP

#include <mpi.h>
#include "comm_data.hpp"
#include "matrix.hpp"
#include "partition.hpp"
#include "par_vector.hpp"

#define STANDARD_PPN 4
#define STANDARD_PROC_LAYOUT 1

/**************************************************************
 *****   CommPkg Class:
 **************************************************************
 ***** This class constructs a parallel communicator, containing
 ***** which messages must be sent/recieved for matrix operations
 *****
 ***** Methods
 ***** -------
 ***** communicate(data_t* values)
 *****    Communicates values to processes, based on underlying
 *****    communication package
 ***** form_col_to_proc(...)
 *****    Maps each column in off_proc_column_map to process 
 *****    on which corresponding values are stored
 **************************************************************/
namespace raptor
{
    class ParCSRMatrix;

    class CommPkg
    {
      public:
        CommPkg(Partition* partition)
        {
            topology = partition->topology;
            topology->num_shared++;
        }
        
        CommPkg(Topology* _topology)
        {
            topology = _topology;
            topology->num_shared++;
        }

        virtual ~CommPkg()
        {
            if (topology)
            {
                if (topology->num_shared)
                {
                    topology->num_shared--;
                }
                else
                {
                    delete topology;
                }
            }
        }

        virtual void update(const aligned_vector<int>& off_proc_col_exists,
                data_t* comm_t = NULL) = 0;

        static MPI_Datatype get_type(aligned_vector<int> buffer)
        {
            return MPI_INT;
        }
        static MPI_Datatype get_type(aligned_vector<double> buffer)
        {
            return MPI_DOUBLE;
        }


        // Matrix Communication
        virtual CSRMatrix* communicate(const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double>& values) = 0;
        virtual CSRMatrix* communicate_T(const aligned_vector<int>& rowptr,
                const aligned_vector<int>& col_indices, const aligned_vector<double>& values, 
                const int n_result_rows) = 0;
        virtual CSRMatrix* communicate(const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices) = 0;
        virtual CSRMatrix* communicate_T(const aligned_vector<int>& rowptr,
                const aligned_vector<int>& col_indices, const int n_result_rows) = 0;

        CSRMatrix* communicate(ParCSRMatrix* A);
        CSRMatrix* communicate(CSRMatrix* A)
        {
            return communicate(A->idx1, A->idx2, A->vals);
        }
        CSRMatrix* communicate_T(CSRMatrix* A)
        {
            return communicate_T(A->idx1, A->idx2, A->vals,
                    A->n_rows);
        }

        // Vector Communication
        aligned_vector<double>& communicate(ParVector& v);
        void init_comm(ParVector& v);

        // Standard Communication
        template<typename T>
        aligned_vector<T>& communicate(const aligned_vector<T>& values)
        {  
            return communicate(values.data());
        }
        template<typename T>
        void init_comm(const aligned_vector<T>& values)
        {
            init_comm(values.data());
        }
        template<typename T> void init_comm(const T* values);
        template<typename T> aligned_vector<T>& complete_comm();
        template<typename T> aligned_vector<T>& communicate(const T* values);
        virtual void init_double_comm(const double* values) = 0;
        virtual void init_int_comm(const int* values) = 0;
        virtual aligned_vector<double>& complete_double_comm() = 0;
        virtual aligned_vector<int>& complete_int_comm() = 0;

        // Transpose Communication
        template<typename T, typename U>
        void communicate_T(const aligned_vector<T>& values, aligned_vector<U>& result,
                std::function<U(U, T)> result_func = {})
        {  
            communicate_T(values.data(), result, result_func);
        }
        template<typename T>
        void communicate_T(const aligned_vector<T>& values)
        {  
            communicate_T(values.data());
        }
        template<typename T>
        void init_comm_T(const aligned_vector<T>& values)
        {
            init_comm_T(values.data());
        }
        template<typename T> void init_comm_T(const T* values);
        template<typename T, typename U> void complete_comm_T(aligned_vector<U>& result,
                std::function<U(U, T)> result_func = {});
        template<typename T> void complete_comm_T();
        template<typename T, typename U> void communicate_T(const T* values, 
                aligned_vector<U>& result, std::function<U(U, T)> result_func = {});
        template<typename T> void communicate_T(const T* values);
        virtual void init_double_comm_T(const double* values) = 0;
        virtual void init_int_comm_T(const int* values) = 0;
        virtual void complete_double_comm_T(aligned_vector<double>& result,
                std::function<double(double, double)> result_func = {}) = 0;
        virtual void complete_double_comm_T(aligned_vector<int>& result,
                std::function<int(int, double)> result_func = {}) = 0;
        virtual void complete_int_comm_T(aligned_vector<int>& result,
                std::function<int(int, int)> result_func = {}) = 0;
        virtual void complete_int_comm_T(aligned_vector<double>& result,
                std::function<double(double, int)> result_func = {}) = 0;
        virtual void complete_double_comm_T() = 0;
        virtual void complete_int_comm_T() = 0;

        // Helper methods
        template <typename T> aligned_vector<T>& get_recv_buffer();
        virtual aligned_vector<double>& get_double_recv_buffer() = 0;
        virtual aligned_vector<int>& get_int_recv_buffer() = 0;

        // Class Variables
        Topology* topology;
    };


    /**************************************************************
    *****   ParComm Class
    **************************************************************
    ***** This class constructs a standard parallel communicator: 
    ***** which messages must be sent/recieved for matrix operations
    *****
    ***** Attributes
    ***** -------------
    ***** num_sends : index_t
    *****    Number of messages this process must send during 
    *****    matrix operations
    ***** num_recvs : index_t
    *****    Number of messages this process will recv during
    *****    matrix operations
    ***** size_sends : index_t 
    *****    Total number of elements this process sends in all
    *****    messages
    ***** size_recvs : index_t 
    *****    Total number of elements this process recvs from
    *****    all messages
    ***** send_procs : aligned_vector<int>
    *****    Distant processes messages are to be sent to
    ***** send_row_starts : aligned_vector<int>
    *****    Pointer to first position in send_row_indices
    *****    that a given process will send.
    ***** send_row_indices : aligned_vector<int> 
    *****    The indices of values that must be sent to each
    *****    process in send_procs
    ***** recv_procs : aligned_vector<int>
    *****    Distant processes messages are to be recvd from
    ***** recv_col_starts : aligned_vector<int>
    *****    Pointer to first column recvd from each process
    *****    in recv_procs
    ***** col_to_proc : aligned_vector<int>
    *****    Maps each local column in the off-diagonal block
    *****    to the process that holds corresponding data
    **************************************************************/
    class ParComm : public CommPkg
    {
      public:
        /**************************************************************
        *****   ParComm Class Constructor
        **************************************************************
        ***** Initializes an empty ParComm, setting send and recv
        ***** sizes to 0
        *****
        ***** Parameters
        ***** -------------
        ***** _key : int (optional)
        *****    Tag to be used in MPI Communication (default 0)
        **************************************************************/
        ParComm(Partition* partition, int _key = 0, 
                MPI_Comm _comm = MPI_COMM_WORLD) : CommPkg(partition)
        {
            mpi_comm = _comm;
            key = _key;
            send_data = new CommData();
            recv_data = new CommData();
        }
        ParComm(Topology* topology, int _key = 0, 
                MPI_Comm _comm = MPI_COMM_WORLD) : CommPkg(topology)
        {
            mpi_comm = _comm;
            key = _key;
            send_data = new CommData();
            recv_data = new CommData();
        }

        /**************************************************************
        *****   ParComm Class Constructor
        **************************************************************
        ***** Initializes a ParComm object based on the off_proc Matrix
        *****
        ***** Parameters
        ***** -------------
        ***** off_proc_column_map : aligned_vector<int>&
        *****    Maps local off_proc columns indices to global
        ***** _key : int (optional)
        *****    Tag to be used in MPI Communication (default 9999)
        **************************************************************/
        ParComm(Partition* partition,
                const aligned_vector<int>& off_proc_column_map,
                int _key = 9999,
                MPI_Comm comm = MPI_COMM_WORLD,
                data_t* comm_t = NULL) : CommPkg(partition)
        {
            mpi_comm = comm;
            init_par_comm(partition, off_proc_column_map, _key, comm, comm_t);
        }

        ParComm(Partition* partition,
                const aligned_vector<int>& off_proc_column_map,
                const aligned_vector<int>& on_proc_column_map,
                int _key = 9999, 
                MPI_Comm comm = MPI_COMM_WORLD,
                data_t* comm_t = NULL) : CommPkg(partition)
        {
            mpi_comm = comm;
            int idx;
            int ctr = 0;
            aligned_vector<int> part_col_to_new;

            init_par_comm(partition, off_proc_column_map, _key, comm, comm_t);
            
            if (partition->local_num_cols)
            {
                part_col_to_new.resize(partition->local_num_cols, -1);
            }
            for (aligned_vector<int>::const_iterator it = on_proc_column_map.begin();
                    it != on_proc_column_map.end(); ++it)
            {
                part_col_to_new[*it - partition->first_local_col] = ctr++;
            }

            for (int i = 0; i < send_data->size_msgs; i++)
            {
                idx = send_data->indices[i];
                send_data->indices[i] = part_col_to_new[idx];
                assert(part_col_to_new[idx] >= 0);
            }
	    
        }

        void init_par_comm(Partition* partition,
                const aligned_vector<int>& off_proc_column_map,
                int _key, MPI_Comm comm, data_t* comm_t = NULL)
        {
            // Get MPI Information
            int rank, num_procs;
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &num_procs);

            // Initialize class variables
            key = _key;
            send_data = new CommData();
            recv_data = new CommData();

            // Declare communication variables
            int send_start, send_end;
            int proc, prev_proc;
            int count;
            int tag = 12345;  // TODO -- switch this to key?
            int off_proc_num_cols = off_proc_column_map.size();
            MPI_Status recv_status;

            aligned_vector<int> off_proc_col_to_proc(off_proc_num_cols);
            aligned_vector<int> tmp_send_buffer;

            partition->form_col_to_proc(off_proc_column_map, off_proc_col_to_proc);

            // Determine processes columns are received from,
            // and adds corresponding messages to recv data.
            // Assumes columns are partitioned across processes
            // in contiguous blocks, and are sorted
            if (off_proc_num_cols)
            {
                prev_proc = off_proc_col_to_proc[0];
                int prev_idx = 0;
                for (int i = 1; i < off_proc_num_cols; i++)
                {
                    proc = off_proc_col_to_proc[i];
                    if (proc != prev_proc)
                    {
                        recv_data->add_msg(prev_proc, i - prev_idx);
                        prev_proc = proc;
                        prev_idx = i;
                    }
                }
                recv_data->add_msg(prev_proc, off_proc_num_cols - prev_idx);
                recv_data->finalize();
            }

            // For each process I recv from, send the global column indices
            // for which I must recv corresponding rows 
            if (recv_data->size_msgs)
            {
                tmp_send_buffer.resize(recv_data->size_msgs);
            }

            if (comm_t) *comm_t -= MPI_Wtime();

            for (int i = 0; i < recv_data->num_msgs; i++)
            {
                proc = recv_data->procs[i];
                send_start = recv_data->indptr[i];
                send_end = recv_data->indptr[i+1];
                for (int j = send_start; j < send_end; j++)
                {
                    tmp_send_buffer[j] = off_proc_column_map[j];
                }
                MPI_Issend(&(tmp_send_buffer[send_start]), send_end - send_start, MPI_INT, 
                        proc, tag, comm, &(recv_data->requests[i]));
            }

            // Determine which processes to which I send messages,
            // and what vector indices to send to each.
            // Receive any messages, regardless of source (which is unknown)
            int finished, msg_avail;
            MPI_Request barrier_request;
	        if (recv_data->num_msgs)
	        {
            	MPI_Testall(recv_data->num_msgs, recv_data->requests.data(), &finished,
                        MPI_STATUSES_IGNORE);
                while (!finished)
                {
                    MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &msg_avail, &recv_status);
                    if (msg_avail)
                    {
                        MPI_Get_count(&recv_status, MPI_INT, &count);
                        proc = recv_status.MPI_SOURCE;
                        int recvbuf[count];
                        MPI_Recv(recvbuf, count, MPI_INT, proc, tag, comm, &recv_status);
                        for (int i = 0; i < count; i++)
                        {
                            recvbuf[i] -= partition->first_local_col;
                        }
                        send_data->add_msg(proc, count, recvbuf);
                    }
                    MPI_Testall(recv_data->num_msgs, recv_data->requests.data(), &finished,
                            MPI_STATUSES_IGNORE);
                }
	        }
            MPI_Ibarrier(comm, &barrier_request);
            MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
            while (!finished)
            {
                MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &msg_avail, &recv_status);
                if (msg_avail)
                {
                    MPI_Get_count(&recv_status, MPI_INT, &count);
                    proc = recv_status.MPI_SOURCE;
                    int recvbuf[count];
                    MPI_Recv(recvbuf, count, MPI_INT, proc, tag, comm, &recv_status);
                    for (int i = 0; i < count; i++)
                    {
                        recvbuf[i] -= partition->first_local_col;
                    }
                    send_data->add_msg(proc, count, recvbuf);
                }
                MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
            }

            if (comm_t) *comm_t += MPI_Wtime();

            if (send_data->num_msgs)
            {
                send_data->finalize();
            }
        }

        ParComm(ParComm* comm) : CommPkg(comm->topology)
        {
            mpi_comm = comm->mpi_comm;
            send_data = new CommData(comm->send_data);
            recv_data = new CommData(comm->recv_data);
            key = comm->key;
        }

        ParComm(ParComm* comm, const aligned_vector<int>& off_proc_col_to_new,
                data_t* comm_t = NULL)
            : CommPkg(comm->topology)
        {
            mpi_comm = comm->mpi_comm;
            bool comm_proc;
            int proc, start, end;
            int idx, new_idx;

            send_data = new CommData();
            recv_data = new CommData();

            if (comm == NULL)
            {
                key = 0;
                return;
            }
            key = comm->key;

            init_off_proc_new(comm, off_proc_col_to_new, comm_t);
        }
        
        ParComm(ParComm* comm, const aligned_vector<int>& on_proc_col_to_new,
                const aligned_vector<int>& off_proc_col_to_new, data_t* comm_t = NULL) 
            : CommPkg(comm->topology)
        {
            mpi_comm = comm->mpi_comm;
            bool comm_proc;
            int proc, start, end;
            int idx, new_idx;

            send_data = new CommData();
            recv_data = new CommData();

            if (comm == NULL)
            {
                key = 0;
                return;
            }
            key = comm->key;
      
            init_off_proc_new(comm, off_proc_col_to_new, comm_t);

            for (int i = 0; i < send_data->size_msgs; i++)
            {
                idx = send_data->indices[i];
                new_idx = on_proc_col_to_new[idx];
                if (new_idx != -1)
                {
                    send_data->indices[i] = new_idx;
                }
            }
        }


        void init_off_proc_new(ParComm* comm, const aligned_vector<int>& off_proc_col_to_new,
                data_t* comm_t = NULL)
        {
            bool comm_proc, comm_idx;
            int proc, start, end;
            int idx, new_idx, ctr;
            int idx_start, idx_end;

            if (comm_t) *comm_t -= MPI_Wtime();
            comm->communicate_T(off_proc_col_to_new);
            if (comm_t) *comm_t += MPI_Wtime();

            if (comm->recv_data->indptr_T.size())
            {
                recv_data->indptr_T.push_back(0);
                for (int i = 0; i < comm->recv_data->num_msgs; i++)
                {
                    comm_proc = false;
                    proc = comm->recv_data->procs[i];
                    start = comm->recv_data->indptr[i];
                    end = comm->recv_data->indptr[i+1];
                    for (int j = start; j < end; j++)
                    {
                        comm_idx = false;
                        idx_start = comm->recv_data->indptr_T[j];
                        idx_end = comm->recv_data->indptr_T[j+1];
                        for (int k = idx_start; k < idx_end; k++)
                        {
                            idx = comm->recv_data->indices[k];
                            new_idx = off_proc_col_to_new[idx];
                            if (new_idx != -1)
                            {
                                comm_proc = true;
                                comm_idx = true;
                                recv_data->indices.push_back(new_idx);
                            }
                        }
                        if (comm_idx)
                        {
                            recv_data->indptr_T.push_back(recv_data->indices.size());
                        }
                    }
                    if (comm_proc)
                    {
                        recv_data->procs.push_back(proc);
                        recv_data->indptr.push_back(recv_data->indptr_T.size() - 1);
                    }
                }
                recv_data->size_msgs = recv_data->indptr_T.size() - 1;
            }
            else if (comm->recv_data->indices.size())
            {
                for (int i = 0; i < comm->recv_data->num_msgs; i++)
                {
                    comm_proc = false;
                    proc = comm->recv_data->procs[i];
                    start = comm->recv_data->indptr[i];
                    end = comm->recv_data->indptr[i+1];
                    for (int j = start; j < end; j++)
                    {
                        idx = comm->recv_data->indices[j];
                        new_idx = off_proc_col_to_new[idx];
                        if (new_idx != -1)
                        {
                            comm_proc = true;
                            recv_data->indices.push_back(new_idx);
                        }
                    }
                    if (comm_proc)
                    {
                        recv_data->procs.push_back(proc);
                        recv_data->indptr.push_back(recv_data->indices.size());
                    }
                }
                recv_data->size_msgs = recv_data->indices.size();
            }
            else
            {
                recv_data->size_msgs = 0;
                for (int i = 0; i < comm->recv_data->num_msgs; i++)
                {
                    comm_proc = false;
                    proc = comm->recv_data->procs[i];
                    start = comm->recv_data->indptr[i];
                    end = comm->recv_data->indptr[i+1];
                    for (int j = start; j < end; j++)
                    {
                        idx = j;
                        new_idx = off_proc_col_to_new[idx];
                        if (new_idx != -1)
                        {
                            comm_proc = true;
                            recv_data->size_msgs++;
                        }
                    }
                    if (comm_proc)
                    {
                        recv_data->procs.push_back(proc);
                        recv_data->indptr.push_back(recv_data->size_msgs);
                    }
                }
            }
            recv_data->num_msgs = recv_data->procs.size();
            recv_data->finalize();

            for (int i = 0; i < comm->send_data->num_msgs; i++)
            {
                comm_proc = false;
                proc = comm->send_data->procs[i];
                start = comm->send_data->indptr[i];
                end = comm->send_data->indptr[i+1];
                for (int j = start; j < end; j++)
                {
                    if (comm->send_data->int_buffer[j] != -1)
                    {
                        comm_proc = true;
                        send_data->indices.push_back(comm->send_data->indices[j]);
                    }
                }
                if (comm_proc)
                {
                    send_data->procs.push_back(proc);
                    send_data->indptr.push_back(send_data->indices.size());
                }
            }
            send_data->num_msgs = send_data->procs.size();
            send_data->size_msgs = send_data->indices.size();
            send_data->finalize();
        }

        /**************************************************************
        *****   ParComm Class Destructor
        **************************************************************
        ***** 
        **************************************************************/
        ~ParComm()
        {
            delete send_data;
            delete recv_data;
        }

        void update(const aligned_vector<int>& off_proc_col_exists, 
                data_t* comm_t = NULL)
        {
            int ctr, send_ctr;
            int start, end;
            int idx;

            if (comm_t) *comm_t -= MPI_Wtime();
            communicate_T(off_proc_col_exists);
            if (comm_t) *comm_t += MPI_Wtime();

            aligned_vector<int>& send_exists = send_data->int_buffer;

            // Update recv_data
            ctr = 0;
            send_ctr = 0;
            start = recv_data->indptr[0];
            for (int i = 0; i < recv_data->num_msgs; i++)
            {
                end = recv_data->indptr[i+1];
                for (int j = start; j < end; j++)
                {
                    if (recv_data->indices.size())
                    {
                        idx = recv_data->indices[j];
                        if (off_proc_col_exists[idx])
                        {
                            recv_data->indices[ctr++] = idx;
                        }
                    }
                    else
                    {
                        if (off_proc_col_exists[j])
                        {
                            ctr++;
                        }
                    }
                }
                if (ctr > start)
                {
                    recv_data->procs[send_ctr] = recv_data->procs[i];
                    recv_data->indptr[++send_ctr] = ctr;
                }
                start = end;
            }
            recv_data->num_msgs = send_ctr;
            recv_data->size_msgs = ctr;
            
            recv_data->indptr.resize(recv_data->num_msgs+1);
            if (recv_data->num_msgs)
            {
                recv_data->procs.resize(recv_data->num_msgs);
                recv_data->requests.resize(recv_data->num_msgs);
                if (recv_data->indices.size())
                {
                    recv_data->indices.resize(recv_data->size_msgs);
                }
                recv_data->buffer.resize(recv_data->size_msgs);
                recv_data->int_buffer.resize(recv_data->size_msgs);
            }

            // Update send_data
            ctr = 0;
            send_ctr = 0;
            start = send_data->indptr[0];
            for (int i = 0; i < send_data->num_msgs; i++)
            {
                end = send_data->indptr[i+1];
                for (int j = start; j < end; j++)
                {
                    if (send_exists[j])
                    {
                        ctr++;
                    }
                }
                if (ctr > start)
                {
                    send_data->procs[send_ctr] = send_data->procs[i];
                    send_data->indptr[++send_ctr] = ctr;
                }
                start = end;
            }
            send_data->num_msgs = send_ctr;
            send_data->size_msgs = ctr;
            send_data->indptr.resize(send_data->num_msgs+1);
            if (send_data->num_msgs)
            {
                send_data->procs.resize(send_data->num_msgs);
                send_data->requests.resize(send_data->num_msgs);
                send_data->indices.resize(send_data->size_msgs);
                send_data->buffer.resize(send_data->size_msgs);
                send_data->int_buffer.resize(send_data->size_msgs);
            }
        }

        // Standard Communication
        void init_double_comm(const double* values)
        {
            initialize(values);
        }
        void init_int_comm(const int* values)
        {
            initialize(values);
        }
        aligned_vector<double>& complete_double_comm()
        {
            return complete<double>();
        }
        aligned_vector<int>& complete_int_comm()
        {
            return complete<int>();
        }
        template<typename T>
        aligned_vector<T>& communicate(const aligned_vector<T>& values)
        {
            return CommPkg::communicate(values.data());
        }
        template<typename T>
        aligned_vector<T>& communicate(const T* values)
        {
            return CommPkg::communicate(values);
        }

        template<typename T>
        void initialize(const T* values)
        {
            int start, end;
            int proc;

            aligned_vector<T>& sendbuf = send_data->get_buffer<T>();
            aligned_vector<T>& recvbuf = recv_data->get_buffer<T>();
            MPI_Datatype type = get_type(sendbuf);

            for (int i = 0; i < send_data->num_msgs; i++)
            {
                proc = send_data->procs[i];
                start = send_data->indptr[i];
                end = send_data->indptr[i+1];
                for (int j = start; j < end; j++)
                {
                    sendbuf[j] = values[send_data->indices[j]];
                }
                MPI_Isend(&(sendbuf[start]), end - start, type,
                        proc, key, mpi_comm, &(send_data->requests[i]));
            }
            for (int i = 0; i < recv_data->num_msgs; i++)
            {
                proc = recv_data->procs[i];
                start = recv_data->indptr[i];
                end = recv_data->indptr[i+1];
                MPI_Irecv(&(recvbuf[start]), end - start, type,
                        proc, key, mpi_comm, &(recv_data->requests[i]));
            }
        }

        template<typename T>
        aligned_vector<T>& complete()
        {
            if (send_data->num_msgs)
            {
                MPI_Waitall(send_data->num_msgs, send_data->requests.data(), MPI_STATUS_IGNORE);
            }

            if (recv_data->num_msgs)
            {
                MPI_Waitall(recv_data->num_msgs, recv_data->requests.data(), MPI_STATUS_IGNORE);
            }

            key++;

            return get_recv_buffer<T>();
        }

        // Transpose Communication
        void init_double_comm_T(const double* values)
        {
            initialize_T(values);
        }
        void init_int_comm_T(const int* values)
        {
            initialize_T(values);
        }
        void complete_double_comm_T(aligned_vector<double>& result,
                std::function<double(double, double)> result_func = {})
        {
            complete_T<double>(result, result_func);
        }
        void complete_double_comm_T(aligned_vector<int>& result,
                std::function<int(int, double)> result_func = {})
        {
            complete_T<double>(result, result_func);
        }
        void complete_int_comm_T(aligned_vector<double>& result,
                std::function<double(double, int)> result_func = {})
        {
            complete_T<int>(result, result_func);
        }
        void complete_int_comm_T(aligned_vector<int>& result,
                std::function<int(int, int)> result_func = {})
        {
            complete_T<int>(result, result_func);
        }
        void complete_double_comm_T()
        {
            complete_T<double>();
        }
        void complete_int_comm_T()
        {
            complete_T<int>();
        }
        template<typename T, typename U>
        void communicate_T(const aligned_vector<T>& values, aligned_vector<U>& result,
                std::function<U(U, T)> result_func = {})
        {
            CommPkg::communicate_T(values.data(), result, result_func);
        }
        template<typename T, typename U>
        void communicate_T(const T* values, aligned_vector<U>& result,
                std::function<U(U, T)> result_func = {})
        {
            CommPkg::communicate_T(values, result, result_func);
        }
        template<typename T>
        void communicate_T(const aligned_vector<T>& values)
        {
            CommPkg::communicate_T(values.data());
        }
        template<typename T>
        void communicate_T(const T* values)
        {
            CommPkg::communicate_T(values);
        }

        template<typename T>
        void initialize_T(const T* values)
        {
            int start, end;
            int proc, idx;
            aligned_vector<T>& sendbuf = send_data->get_buffer<T>();
            aligned_vector<T>& recvbuf = recv_data->get_buffer<T>();
            MPI_Datatype type = get_type(sendbuf);

            if (recv_data->indptr_T.size())
            {
                int idx_start, idx_end;
                T val;
                for (int i = 0; i < recv_data->num_msgs; i++)
                {
                    proc = recv_data->procs[i];
                    start = recv_data->indptr[i];
                    end = recv_data->indptr[i+1];
                    for (int j = start; j < end; j++)
                    {
                        idx_start = recv_data->indptr_T[j];
                        idx_end = recv_data->indptr_T[j+1];
                        val = 0;
                        for (int k = idx_start; k < idx_end; k++)
                        {
                            val += values[recv_data->indices[k]];
                        }
                        recvbuf[j] = val;
                    }
                    MPI_Isend(&(recvbuf[start]), end - start, type,
                            proc, key, mpi_comm, &(recv_data->requests[i]));
                }
            }
            else if (recv_data->indices.size())
            {
                for (int i = 0; i < recv_data->num_msgs; i++)
                {
                    proc = recv_data->procs[i];
                    start = recv_data->indptr[i];
                    end = recv_data->indptr[i+1];
                    for (int j = start; j < end; j++)
                    {
                        idx = recv_data->indices[j];
                        recvbuf[j] = values[idx];
                    }
                    MPI_Isend(&(recvbuf[start]), end - start, type,
                            proc, key, mpi_comm, &(recv_data->requests[i]));
                }
            }
            else
            {
                for (int i = 0; i < recv_data->num_msgs; i++)
                {
                    proc = recv_data->procs[i];
                    start = recv_data->indptr[i];
                    end = recv_data->indptr[i+1];
                    for (int j = start; j < end; j++)
                    {
                        idx = j;
                        recvbuf[j] = values[idx];
                    }
                    MPI_Isend(&(recvbuf[start]), end - start, type,
                            proc, key, mpi_comm, &(recv_data->requests[i]));
                }
            }
            for (int i = 0; i < send_data->num_msgs; i++)
            {
                proc = send_data->procs[i];
                start = send_data->indptr[i];
                end = send_data->indptr[i+1];
                MPI_Irecv(&(sendbuf[start]), end - start, type,
                        proc, key, mpi_comm, &(send_data->requests[i]));
            }
        }

        template<typename T, typename U>
        void complete_T(aligned_vector<U>& result, std::function<U(U, T)> result_func = {})
        {
            complete_T<T>();

            int idx;
            aligned_vector<T>& sendbuf = send_data->get_buffer<T>();

            if (result_func)
            {
                for (int i = 0; i < send_data->size_msgs; i++)
                {
                    idx = send_data->indices[i];
                    result[idx]  = result_func(result[idx], sendbuf[i]);
                }
            }
            else
            {
                for (int i = 0; i < send_data->size_msgs; i++)
                {
                    idx = send_data->indices[i];
                    result[idx] += sendbuf[i];
                }
            }
        }

        template<typename T>
        void complete_T()
        {
            if (send_data->num_msgs)
            {
                MPI_Waitall(send_data->num_msgs, send_data->requests.data(), MPI_STATUSES_IGNORE);
            }

            if (recv_data->num_msgs)
            {
                MPI_Waitall(recv_data->num_msgs, recv_data->requests.data(), MPI_STATUSES_IGNORE);
            }
            key++;
        }

        // Conditional communication
        template <typename T>
        aligned_vector<T>& conditional_comm(const aligned_vector<T>& vals,  
                const aligned_vector<int>& states, 
                const aligned_vector<int>& off_proc_states,
                std::function<bool(int)> compare_func)
        {
            int proc, start, end;
            int idx, size;
            int ctr, prev_ctr;
            int n_sends, n_recvs;
            int key = 325493;

            aligned_vector<T>& sendbuf = send_data->get_buffer<T>();
            aligned_vector<T>& recvbuf = recv_data->get_buffer<T>();
            MPI_Datatype type = get_type(sendbuf);

            n_sends = 0;
            ctr = 0;
            prev_ctr = 0;
            for (int i = 0; i < send_data->num_msgs; i++)
            {
                proc = send_data->procs[i];
                start = send_data->indptr[i];
                end = send_data->indptr[i+1];
                for (int j = start; j < end; j++)
                {
                    idx = send_data->indices[j];
                    if (compare_func(states[idx]))
                    {
                        sendbuf[ctr++] = vals[idx];
                    }
                }
                size = ctr - prev_ctr;
                if (size)
                {
                    MPI_Isend(&(sendbuf[prev_ctr]), size, type, 
                            proc, key, mpi_comm, &(send_data->requests[n_sends++]));
                    prev_ctr = ctr;
                }
            }
            

            n_recvs = 0;
            ctr = 0;
            prev_ctr = 0;
            for (int i = 0; i < recv_data->num_msgs; i++)
            {
                proc = recv_data->procs[i];
                start = recv_data->indptr[i];
                end = recv_data->indptr[i+1];
                for (int j = start; j < end; j++)
                {
                    idx = j;

                    if (compare_func(off_proc_states[idx]))
                    {
                        ctr++;
                    }
                }
                size = ctr - prev_ctr;
                if (size)
                {
                    MPI_Irecv(&(recvbuf[prev_ctr]), size, type,
                            proc, key, mpi_comm, &(recv_data->requests[n_recvs++]));
                    prev_ctr = ctr;
                }
            }

            if (n_sends)
            {
                MPI_Waitall(n_sends, send_data->requests.data(), MPI_STATUSES_IGNORE);
            }
            if (n_recvs)
            {
                MPI_Waitall(n_recvs, recv_data->requests.data(), MPI_STATUSES_IGNORE);
            }

            ctr--;
            for (int i = recv_data->size_msgs - 1; i >= 0; i--)
            {
                if (compare_func(off_proc_states[i]))
                {
                    recvbuf[i] = recvbuf[ctr--];
                }
                else
                {
                    recvbuf[i] = 0.0;
                }
            }

            return recvbuf;
        }

        template <typename T, typename U>
        void conditional_comm_T(const aligned_vector<T>& vals,  
                const aligned_vector<int>& states, 
                const aligned_vector<int>& off_proc_states,
                std::function<bool(int)> compare_func,
                aligned_vector<U>& result, 
                std::function<U(U, T)> result_func)
        {
            int proc, start, end;
            int idx, size;
            int ctr, prev_ctr;
            int n_sends, n_recvs;

            aligned_vector<T>& sendbuf = send_data->get_buffer<T>();
            aligned_vector<T>& recvbuf = recv_data->get_buffer<T>();
            MPI_Datatype type = get_type(sendbuf);
            int key = 453246;

            n_sends = 0;
            ctr = 0;
            prev_ctr = 0;
            for (int i = 0; i < recv_data->num_msgs; i++)
            {
                proc = recv_data->procs[i];
                start = recv_data->indptr[i];
                end = recv_data->indptr[i+1];
                for (int j = start; j < end; j++)
                {
                    if (compare_func(off_proc_states[j]))
                    {
                        recvbuf[ctr++] = vals[j];
                    }
                }
                size = ctr - prev_ctr;
                if (size)
                {
                    MPI_Issend(&(recvbuf[prev_ctr]), size, type, 
                            proc, key, mpi_comm, &(recv_data->requests[n_sends++]));
                    prev_ctr = ctr;
                }
            }

            n_recvs = 0;
            ctr = 0;
            prev_ctr = 0;
            for (int i = 0; i < send_data->num_msgs; i++)
            {
                proc = send_data->procs[i];
                start = send_data->indptr[i];
                end = send_data->indptr[i+1];
                for (int j = start; j < end; j++)
                {
                    idx = send_data->indices[j];
                    if (compare_func(states[idx]))
                    {
                        ctr++;
                    }
                }
                size = ctr - prev_ctr;
                if (size)
                {
                    MPI_Irecv(&(sendbuf[prev_ctr]), size, type,
                            proc, key, mpi_comm, &(send_data->requests[n_recvs++]));
                    prev_ctr = ctr;
                }
            }

            if (n_sends)
            {
                MPI_Waitall(n_sends, recv_data->requests.data(), MPI_STATUSES_IGNORE);
            }
            if (n_recvs)
            {
                MPI_Waitall(n_recvs, send_data->requests.data(), MPI_STATUSES_IGNORE);
            }

            ctr = 0;
            for (int i = 0; i < send_data->size_msgs; i++)
            {
                idx = send_data->indices[i];
                if (compare_func(states[idx]))
                {
                    result[idx] = result_func(result[idx], sendbuf[ctr++]);
                }
            }
        }


        // Matrix Communication
        CSRMatrix* communicate(const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double>& values);
        CSRMatrix* communicate_T(const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double>& values, 
                const int n_result_rows);
        CSRMatrix* communicate(const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices);
        CSRMatrix* communicate_T(const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const int n_result_rows);
        CSRMatrix* communicate(ParCSRMatrix* A)
        {
            return CommPkg::communicate(A);
        }
        CSRMatrix* communicate(CSRMatrix* A)
        {
            return CommPkg::communicate(A);
        }
        CSRMatrix* communicate_T(CSRMatrix* A)
        {
            return CommPkg::communicate_T(A);
        }


        // Vector Communication
        aligned_vector<double>& communicate(ParVector& v)
        {
            return CommPkg::communicate(v);
        }
        void init_comm(ParVector& v)
        {
            CommPkg::init_comm(v);
        }

        // Helper Methods
        aligned_vector<double>& get_double_recv_buffer()
        {
            return recv_data->buffer;
        }
        aligned_vector<int>& get_int_recv_buffer()
        {
            return recv_data->int_buffer;
        }
        aligned_vector<double>& get_double_send_buffer()
        {
            return send_data->buffer;
        }
        aligned_vector<int>& get_int_send_buffer()
        {
            return send_data->int_buffer;
        }


        int key;
        CommData* send_data;
        CommData* recv_data;
        MPI_Comm mpi_comm;
    };



    /**************************************************************
    *****   TAPComm Class
    **************************************************************
    ***** This class constructs a topology-aware parallel communicator: 
    ***** which messages must be sent/recieved for matrix operations,
    ***** using topology-aware methods to limit the number and size
    ***** of inter-node messages
    *****
    ***** Attributes
    ***** -------------
    ***** local_S_par_comm : ParComm*
    *****    Parallel communication package for sending data that originates
    *****    on rank to other processes local to node, before inter-node
    *****    communication occurs.
    ***** local_R_par_comm : ParComm*
    *****    Parallel communication package for redistributing previously
    *****    received values (from inter-node communication step) to 
    *****    processes local to rank which need said values
    ***** local_L_par_comm : ParComm* 
    *****    Parallel communication package for communicating values
    *****    that both originate and have a final destination on node
    *****    (fully intra-node communication)
    ***** global_par_comm : ParComm*
    *****    Parallel communication package for sole inter-node step.
    ***** recv_buffer : Vector
    *****    Combination of local_L_par_comm and local_R_par_comm
    *****    recv buffers, ordered to match off_proc_column_map
    ***** Partition* partition
    *****    Partition, holding information about topology
    **************************************************************/
    class TAPComm : public CommPkg
    {
        public:

        TAPComm(Partition* partition) : CommPkg(partition)
        {
            local_S_par_comm = new ParComm(partition, 2345, partition->topology->local_comm);
            local_R_par_comm = new ParComm(partition, 3456, partition->topology->local_comm);
            local_L_par_comm = new ParComm(partition, 4567, partition->topology->local_comm);
            global_par_comm = new ParComm(partition, 5678, MPI_COMM_WORLD);
        }


        /**************************************************************
        *****   TAPComm Class Constructor
        **************************************************************
        ***** Initializes a TAPComm for a matrix without contiguous
        ***** row-wise partitions across processes.  Instead, each
        ***** process holds a random assortment of rows. 
        *****
        ***** Parameters
        ***** -------------
        ***** off_proc_column_map : aligned_vector<int>&
        *****    Maps local off_proc columns indices to global
        ***** global_num_cols : int
        *****    Number of global columns in matrix
        ***** local_num_cols : int
        *****    Number of columns local to rank
        **************************************************************/
        TAPComm(Partition* partition, 
                const aligned_vector<int>& off_proc_column_map,
                bool form_S = true,
                MPI_Comm comm = MPI_COMM_WORLD,
                data_t* comm_t = NULL) : CommPkg(partition)
        {
            if (form_S)
            {
                init_tap_comm(partition, off_proc_column_map, comm, comm_t);
            }
            else
            {
                init_tap_comm_simple(partition, off_proc_column_map, comm, comm_t);
            }
        }

        TAPComm(Partition* partition,
                const aligned_vector<int>& off_proc_column_map,
                const aligned_vector<int>& on_proc_column_map,
                bool form_S = true,
                MPI_Comm comm = MPI_COMM_WORLD,
                data_t* comm_t = NULL) : CommPkg(partition)
        {
            aligned_vector<int> on_proc_to_new;
            int on_proc_num_cols = on_proc_column_map.size();
            if (partition->local_num_cols)
            {
                on_proc_to_new.resize(partition->local_num_cols);
                for (int i = 0; i < on_proc_num_cols; i++)
                {
                    on_proc_to_new[on_proc_column_map[i] - partition->first_local_col] = i;
                }
            }

            if (form_S)
            {
                init_tap_comm(partition, off_proc_column_map, comm, comm_t);

                for (aligned_vector<int>::iterator it = local_S_par_comm->send_data->indices.begin();
                        it != local_S_par_comm->send_data->indices.end(); ++it)
                {
                    *it = on_proc_to_new[*it];
                }
            }
            else
            {
                init_tap_comm_simple(partition, off_proc_column_map, comm, comm_t);

                for (aligned_vector<int>::iterator it = global_par_comm->send_data->indices.begin();
                        it != global_par_comm->send_data->indices.end(); ++it)
                {
                    *it = on_proc_to_new[*it];
                }
            }

            for (aligned_vector<int>::iterator it = local_L_par_comm->send_data->indices.begin();
                    it != local_L_par_comm->send_data->indices.end(); ++it)
            {
                *it = on_proc_to_new[*it];
            }
        }

        /**************************************************************
        *****   TAPComm Class Constructor
        **************************************************************
        ***** Create topology-aware communication class from 
        ***** original communication package (which processes rank
        ***** communication which, and what is sent to / recv from
        ***** each process.
        *****
        ***** Parameters
        ***** -------------
        ***** orig_comm : ParComm*
        *****    Existing standard communication package from which
        *****    to form topology-aware communicator
        **************************************************************/
        TAPComm(TAPComm* tap_comm) : CommPkg(tap_comm->topology)
        {
            if (tap_comm->local_S_par_comm)
            {
                local_S_par_comm = new ParComm(tap_comm->local_S_par_comm);
            }

            global_par_comm = new ParComm(tap_comm->global_par_comm);
            local_R_par_comm = new ParComm(tap_comm->local_R_par_comm);
            local_L_par_comm = new ParComm(tap_comm->local_L_par_comm);

            int recv_size = tap_comm->recv_buffer.size();
            if (recv_size)
            {
                recv_buffer.resize(recv_size);
                int_recv_buffer.resize(recv_size);
            }
        }

        TAPComm(TAPComm* tap_comm, const aligned_vector<int>& off_proc_col_to_new, 
                data_t* comm_t = NULL) : CommPkg(tap_comm->topology)
        {
            init_off_proc_new(tap_comm, off_proc_col_to_new, comm_t);
        }

        TAPComm(TAPComm* tap_comm, const aligned_vector<int>& on_proc_col_to_new,
                const aligned_vector<int>& off_proc_col_to_new, 
                data_t* comm_t = NULL) : CommPkg(tap_comm->topology)
        {
            int idx;

            init_off_proc_new(tap_comm, off_proc_col_to_new, comm_t);

            for (int i = 0; i < local_L_par_comm->send_data->size_msgs; i++)
            {
                idx = local_L_par_comm->send_data->indices[i];
                local_L_par_comm->send_data->indices[i] = on_proc_col_to_new[idx];
            }

            if (local_S_par_comm)
            {
                for (int i = 0; i < local_S_par_comm->send_data->size_msgs; i++)
                {
                    idx = local_S_par_comm->send_data->indices[i];
                    local_S_par_comm->send_data->indices[i] = on_proc_col_to_new[idx];
                }
            }
            else
            {
                for (int i = 0; i < global_par_comm->send_data->size_msgs; i++)
                {
                    idx = global_par_comm->send_data->indices[i];
                    global_par_comm->send_data->indices[i] = on_proc_col_to_new[idx];
                }
            }
        }


        void init_off_proc_new(TAPComm* tap_comm, const aligned_vector<int>& off_proc_col_to_new,
                data_t* comm_t = NULL)
        {
            int idx, ctr;
            int start, end;

            local_L_par_comm = new ParComm(tap_comm->local_L_par_comm, off_proc_col_to_new, 
                    comm_t);
            local_R_par_comm = new ParComm(tap_comm->local_R_par_comm, off_proc_col_to_new,
                    comm_t);

            // Create global par comm / update R send indices
            aligned_vector<int> G_to_new(tap_comm->global_par_comm->recv_data->size_msgs, -1);
            ctr = 0;
            for (int i = 0; i < tap_comm->global_par_comm->recv_data->size_msgs; i++)
            {
                start = tap_comm->global_par_comm->recv_data->indptr_T[i];
                end = tap_comm->global_par_comm->recv_data->indptr_T[i+1];
                for (int j = start; j < end; j++)
                {
                    idx = tap_comm->global_par_comm->recv_data->indices[j];
                    if (tap_comm->local_R_par_comm->send_data->int_buffer[idx] != -1)
                    {
                        G_to_new[i] = ctr++;
                        break;
                    }
                }
            }
            for (aligned_vector<int>::iterator it = local_R_par_comm->send_data->indices.begin();
                    it != local_R_par_comm->send_data->indices.end(); ++it)
            {
                *it = G_to_new[*it];
            }
            idx = 0;
            for (aligned_vector<int>::iterator it = tap_comm->local_R_par_comm->send_data->int_buffer.begin();
                    it != tap_comm->local_R_par_comm->send_data->int_buffer.end(); ++it)
            {
                if (*it != -1) *it = idx++;
            }
            global_par_comm = new ParComm(tap_comm->global_par_comm, 
                    tap_comm->local_R_par_comm->send_data->int_buffer, comm_t);



            // create local S / update global send indices
            if (tap_comm->local_S_par_comm)
            {
                aligned_vector<int> S_to_new(tap_comm->local_S_par_comm->recv_data->size_msgs, -1);
                ctr = 0;
                for (int i = 0; i < tap_comm->local_S_par_comm->recv_data->size_msgs; i++)
                {
                    start = tap_comm->local_S_par_comm->recv_data->indptr_T[i];
                    end = tap_comm->local_S_par_comm->recv_data->indptr_T[i+1];
                    for (int j = start; j < end; j++)
                    {
                        idx = tap_comm->local_S_par_comm->recv_data->indices[j];
                        if (tap_comm->global_par_comm->send_data->int_buffer[idx] != -1)
                        {
                            S_to_new[i] = ctr++;
                            break;
                        }
                    }
                }
                for (aligned_vector<int>::iterator it = global_par_comm->send_data->indices.begin();
                        it != global_par_comm->send_data->indices.end(); ++it)
                {
                    *it = S_to_new[*it];
                }
                idx = 0;
                for (aligned_vector<int>::iterator it = tap_comm->global_par_comm->send_data->int_buffer.begin();
                        it != tap_comm->global_par_comm->send_data->int_buffer.end(); ++it)
                {
                    if (*it != -1) *it = idx++;
                }
                local_S_par_comm = new ParComm(tap_comm->local_S_par_comm,
                        tap_comm->global_par_comm->send_data->int_buffer, comm_t);
            }

            // Determine size of final recvs (should be equal to 
            // number of off_proc cols)
            int recv_size = local_R_par_comm->recv_data->size_msgs +
                local_L_par_comm->recv_data->size_msgs;
            if (recv_size)
            {
                // Want a single recv buffer local_R and local_L par_comms
                recv_buffer.resize(recv_size);
                int_recv_buffer.resize(recv_size);
            }        
        }

        /**************************************************************
        *****   ParComm Class Destructor
        **************************************************************
        ***** 
        **************************************************************/
        ~TAPComm()
        {
            delete global_par_comm;
            delete local_S_par_comm;  // May be NULL but can still call delete
            delete local_R_par_comm;
            delete local_L_par_comm;
        }

        void init_tap_comm(Partition* partition,
                const aligned_vector<int>& off_proc_column_map,
                MPI_Comm comm, data_t* comm_t = NULL)
        {
            // Get MPI Information
            int rank, num_procs;
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &num_procs);

            // Initialize class variables
            local_S_par_comm = new ParComm(partition, 2345, partition->topology->local_comm);
            local_R_par_comm = new ParComm(partition, 3456, partition->topology->local_comm);
            local_L_par_comm = new ParComm(partition, 4567, partition->topology->local_comm);
            global_par_comm = new ParComm(partition, 5678, comm);

            // Initialize Variables
            int idx;
            int recv_size;
            aligned_vector<int> off_proc_col_to_proc;
            aligned_vector<int> on_node_column_map;
            aligned_vector<int> on_node_col_to_proc;
            aligned_vector<int> off_node_column_map;
            aligned_vector<int> off_node_col_to_node;
            aligned_vector<int> on_node_to_off_proc;
            aligned_vector<int> off_node_to_off_proc;
            aligned_vector<int> recv_nodes;
            aligned_vector<int> orig_procs;
            aligned_vector<int> node_to_local_proc;

            // Find process on which vector value associated with each column is
            // stored
            partition->form_col_to_proc(off_proc_column_map, off_proc_col_to_proc);

            // Partition off_proc cols into on_node and off_node
            split_off_proc_cols(off_proc_column_map, off_proc_col_to_proc,
                   on_node_column_map, on_node_col_to_proc, on_node_to_off_proc,
                   off_node_column_map, off_node_col_to_node, off_node_to_off_proc);

            // Gather all nodes with which any local process must communication
            form_local_R_par_comm(off_node_column_map, off_node_col_to_node, 
                    orig_procs, comm_t);

            // Find global processes with which rank communications
            form_global_par_comm(orig_procs, comm_t);

            // Form local_S_par_comm: initial distribution of values among local
            // processes, before inter-node communication
            form_local_S_par_comm(orig_procs, comm_t);

            // Adjust send indices (currently global vector indices) to be index 
            // of global vector value from previous recv
            adjust_send_indices(partition->first_local_col);

            // Form local_L_par_comm: fully local communication (origin and
            // destination processes both local to node)
            form_local_L_par_comm(on_node_column_map, on_node_col_to_proc,
                    partition->first_local_col, comm_t);

            // Determine size of final recvs (should be equal to 
            // number of off_proc cols)
            recv_size = local_R_par_comm->recv_data->size_msgs +
                local_L_par_comm->recv_data->size_msgs;
            if (recv_size)
            {
                // Want a single recv buffer local_R and local_L par_comms
                recv_buffer.resize(recv_size);
                int_recv_buffer.resize(recv_size);

                // Map local_R recvs to original off_proc_column_map
                if (local_R_par_comm->recv_data->size_msgs)
                {
                    for (int i = 0; i < local_R_par_comm->recv_data->size_msgs; i++)
                    {
                        idx = local_R_par_comm->recv_data->indices[i];
                        local_R_par_comm->recv_data->indices[i] = off_node_to_off_proc[idx];
                    }
                }


                // Map local_L recvs to original off_proc_column_map
                if (local_L_par_comm->recv_data->size_msgs)
                {
                    for (int i = 0; i < local_L_par_comm->recv_data->size_msgs; i++)
                    {
                        idx = local_L_par_comm->recv_data->indices[i];
                        local_L_par_comm->recv_data->indices[i] = on_node_to_off_proc[idx];
                    }
                }
            }
        }

        void init_tap_comm_simple(Partition* partition,
                const aligned_vector<int>& off_proc_column_map,
                MPI_Comm comm, data_t* comm_t = NULL)
        {
            // Get MPI Information
            int rank, num_procs;
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &num_procs);

            // Initialize class variables
            local_S_par_comm = NULL;
            local_R_par_comm = new ParComm(partition, 3456, partition->topology->local_comm);
            local_L_par_comm = new ParComm(partition, 4567, partition->topology->local_comm);
            global_par_comm = new ParComm(partition, 5678, comm);

            // Initialize Variables
            int idx;
            int recv_size;
            aligned_vector<int> off_proc_col_to_proc;
            aligned_vector<int> on_node_column_map;
            aligned_vector<int> on_node_col_to_proc;
            aligned_vector<int> off_node_column_map;
            aligned_vector<int> off_node_col_to_proc;
            aligned_vector<int> on_node_to_off_proc;
            aligned_vector<int> off_node_to_off_proc;

            // Find process on which vector value associated with each column is
            // stored
            partition->form_col_to_proc(off_proc_column_map, off_proc_col_to_proc);

            // Partition off_proc cols into on_node and off_node
            split_off_proc_cols(off_proc_column_map, off_proc_col_to_proc,
                   on_node_column_map, on_node_col_to_proc, on_node_to_off_proc,
                   off_node_column_map, off_node_col_to_proc, off_node_to_off_proc);

            // Form local recv communicator.  Will recv from local rank
            // corresponding to global rank on which data originates.  E.g. if
            // data is on rank r = (p, n), and my rank is s = (q, m), I will
            // recv data from (p, m).
            form_simple_R_par_comm(off_node_column_map, off_node_col_to_proc, comm_t);

            // Form global par comm.. Will recv from proc on which data
            // originates
            form_simple_global_comm(off_node_col_to_proc, comm_t);

            // Adjust send indices (currently global vector indices) to be
            // index of global vector value from previous recv (only updating
            // local_R to match position in global)
            adjust_send_indices(partition->first_local_col);

            // Form local_L_par_comm: fully local communication (origin and
            // destination processes both local to node)
            form_local_L_par_comm(on_node_column_map, on_node_col_to_proc,
                    partition->first_local_col, comm_t);

            // Determine size of final recvs (should be equal to 
            // number of off_proc cols)
            recv_size = local_R_par_comm->recv_data->size_msgs +
                local_L_par_comm->recv_data->size_msgs;
            if (recv_size)
            {
                // Want a single recv buffer local_R and local_L par_comms
                recv_buffer.resize(recv_size);
                int_recv_buffer.resize(recv_size);

                // Map local_R recvs to original off_proc_column_map
                if (local_R_par_comm->recv_data->size_msgs)
                {
                    for (int i = 0; i < local_R_par_comm->recv_data->size_msgs; i++)
                    {
                        idx = local_R_par_comm->recv_data->indices[i];
                        local_R_par_comm->recv_data->indices[i] = off_node_to_off_proc[idx];
                    }
                }


                // Map local_L recvs to original off_proc_column_map
                if (local_L_par_comm->recv_data->size_msgs)
                {
                    for (int i = 0; i < local_L_par_comm->recv_data->size_msgs; i++)
                    {
                        idx = local_L_par_comm->recv_data->indices[i];
                        local_L_par_comm->recv_data->indices[i] = on_node_to_off_proc[idx];
                    }
                }
            }
        }

        void update(const aligned_vector<int>& off_proc_col_exists,
                data_t* comm_t = NULL)
        {
            // TODO
        }


        // Helper methods for forming TAPComm:
        void split_off_proc_cols(const aligned_vector<int>& off_proc_column_map,
                const aligned_vector<int>& off_proc_col_to_proc,
                aligned_vector<int>& on_node_column_map,
                aligned_vector<int>& on_node_col_to_proc,
                aligned_vector<int>& on_node_to_off_proc,
                aligned_vector<int>& off_node_column_map,
                aligned_vector<int>& off_node_col_to_node,
                aligned_vector<int>& off_node_to_off_proc);
        void form_local_R_par_comm(const aligned_vector<int>& off_node_column_map,
                const aligned_vector<int>& off_node_col_to_node,
                aligned_vector<int>& orig_procs, data_t* comm_t = NULL);
        void form_global_par_comm(aligned_vector<int>& orig_procs, data_t* comm_t = NULL);
        void form_local_S_par_comm(aligned_vector<int>& orig_procs, data_t* comm_t = NULL);
        void adjust_send_indices(const int first_local_col);
        void form_local_L_par_comm(const aligned_vector<int>& on_node_column_map,
                const aligned_vector<int>& on_node_col_to_proc,
                const int first_local_col, data_t* comm_t = NULL);
        void form_simple_R_par_comm(aligned_vector<int>& off_node_column_map,
                aligned_vector<int>& off_node_col_to_proc, data_t* comm_t = NULL);
        void form_simple_global_comm(aligned_vector<int>& off_node_col_to_proc,
                data_t* comm_t = NULL);

        // Class Methods
        void init_double_comm(const double* values)
        {
            initialize(values);
        }
        void init_int_comm(const int* values)
        {
            initialize(values);
        }
        aligned_vector<double>& complete_double_comm()
        {
            return complete<double>();
        }
        aligned_vector<int>& complete_int_comm()
        {
            return complete<int>();
        }
        
        template<typename T>
        aligned_vector<T>& communicate(const aligned_vector<T>& values)
        {
            return CommPkg::communicate<T>(values.data());
        }
        template<typename T>
        aligned_vector<T>& communicate(const T* values)
        {
            return CommPkg::communicate<T>(values);
        }

        template<typename T>
        void initialize(const T* values)
        {
            // Messages with origin and final destination on node
            local_L_par_comm->communicate<T>(values);

            if (local_S_par_comm)
            {
                // Initial redistribution among node
                aligned_vector<T>& S_vals = local_S_par_comm->communicate<T>(values);

                // Begin inter-node communication 
                global_par_comm->initialize(S_vals.data());
            }
            else
            {
                global_par_comm->initialize(values);
            }
        }

        template<typename T>
        aligned_vector<T>& complete()
        {
            // Complete inter-node communication
            aligned_vector<T>& G_vals = global_par_comm->complete<T>();

            // Redistributing recvd inter-node values
            local_R_par_comm->communicate<T>(G_vals.data());

            aligned_vector<T>& recvbuf = get_recv_buffer<T>();
            aligned_vector<T>& R_recvbuf = local_R_par_comm->recv_data->get_buffer<T>();
            aligned_vector<T>& L_recvbuf = local_L_par_comm->recv_data->get_buffer<T>();

            // Add values from L_recv and R_recv to appropriate positions in 
            // Vector recv
            int idx, new_idx;
            int R_recv_size = local_R_par_comm->recv_data->size_msgs;
            int L_recv_size = local_L_par_comm->recv_data->size_msgs;
            for (int i = 0; i < R_recv_size; i++)
            {
                idx = local_R_par_comm->recv_data->indices[i];
                recvbuf[idx] = R_recvbuf[i];
            }

            for (int i = 0; i < L_recv_size; i++)
            {
                idx = local_L_par_comm->recv_data->indices[i];
                recvbuf[idx] = L_recvbuf[i];
            }

            return recvbuf;
        }


        // Transpose Communication
        void init_double_comm_T(const double* values)
        {
            initialize_T(values);
        }
        void init_int_comm_T(const int* values)
        {
            initialize_T(values);
        }
        void complete_double_comm_T(aligned_vector<double>& result,
                std::function<double(double, double)> result_func = {})
        {
            complete_T<double>(result, result_func);
        }        
        void complete_double_comm_T(aligned_vector<int>& result,
                std::function<int(int, double)> result_func = {})
        {
            complete_T<double>(result, result_func);
        }
        void complete_int_comm_T(aligned_vector<double>& result,
                std::function<double(double, int)> result_func = {})
        {
            complete_T<int>(result, result_func);
        }
        void complete_int_comm_T(aligned_vector<int>& result,
                std::function<int(int, int)> result_func = {})
        {
            complete_T<int>(result, result_func);
        }

        void complete_double_comm_T()
        {
            complete_T<double>();
        }
        void complete_int_comm_T()
        {
            complete_T<int>();
        }

        template<typename T, typename U>
        void communicate_T(const aligned_vector<T>& values, aligned_vector<U>& result,
                std::function<U(U, T)> result_func = {})
        {
            CommPkg::communicate_T(values.data(), result, result_func);
        }
        template<typename T, typename U>
        void communicate_T(const T* values, aligned_vector<U>& result,
                std::function<U(U, T)> result_func = {})
        {
            CommPkg::communicate_T(values, result, result_func);
        }
        template<typename T>
        void communicate_T(const aligned_vector<T>& values)
        {
            CommPkg::communicate_T(values.data());
        }
        template<typename T>
        void communicate_T(const T* values)
        {
            CommPkg::communicate_T(values);
        }

        template<typename T>
        void initialize_T(const T* values)
        {
            int idx;

            // Messages with origin and final destination on node
            local_L_par_comm->communicate_T(values);

            // Initial redistribution among node
            local_R_par_comm->communicate_T(values);

            // Begin inter-node communication 
            aligned_vector<T>& R_sendbuf = local_R_par_comm->send_data->get_buffer<T>();
            global_par_comm->init_comm_T(R_sendbuf);
        }

        template<typename T, typename U>
        void complete_T(aligned_vector<U>& result, std::function<U(U, T)> result_func)
        {
            complete_T<T>();
            int idx;
            aligned_vector<T>& L_sendbuf = local_L_par_comm->send_data->get_buffer<T>();

            if (result_func)
            {
                for (int i = 0; i < local_L_par_comm->send_data->size_msgs; i++)
                {
                    idx = local_L_par_comm->send_data->indices[i];
                    result[idx] = result_func(result[idx], L_sendbuf[i]);
                }

                if (local_S_par_comm)
                {
                    aligned_vector<T>& S_sendbuf = local_S_par_comm->send_data->get_buffer<T>();
                    for (int i = 0; i < local_S_par_comm->send_data->size_msgs; i++)
                    {
                        idx = local_S_par_comm->send_data->indices[i];
                        result[idx] = result_func(result[idx], S_sendbuf[i]);
                    }
                }
                else
                {
                    aligned_vector<T>& G_sendbuf = global_par_comm->send_data->get_buffer<T>();
                    for (int i = 0; i < global_par_comm->send_data->size_msgs; i++)
                    {
                        idx = global_par_comm->send_data->indices[i];
                        result[idx] = result_func(result[idx], G_sendbuf[i]);
                    }
                }
            }
            else
            {
                for (int i = 0; i < local_L_par_comm->send_data->size_msgs; i++)
                {
                    idx = local_L_par_comm->send_data->indices[i];
                    result[idx] += L_sendbuf[i];
                }

                if (local_S_par_comm)
                {
                    aligned_vector<T>& S_sendbuf = local_S_par_comm->send_data->get_buffer<T>();
                    for (int i = 0; i < local_S_par_comm->send_data->size_msgs; i++)
                    {
                        idx = local_S_par_comm->send_data->indices[i];
                        result[idx] += S_sendbuf[i];
                    }
                }
                else
                {
                    aligned_vector<T>& G_sendbuf = global_par_comm->send_data->get_buffer<T>();
                    for (int i = 0; i < global_par_comm->send_data->size_msgs; i++)
                    {
                        idx = global_par_comm->send_data->indices[i];
                        result[idx] += G_sendbuf[i];
                    }
                }
            }

        }
        template<typename T>
        void complete_T()
        {
            // Complete inter-node communication
            global_par_comm->complete_comm_T<T>();
    
            if (local_S_par_comm)
            {
                aligned_vector<T>& G_sendbuf = global_par_comm->send_data->get_buffer<T>();
                local_S_par_comm->communicate_T(G_sendbuf);
            }
        }


        // Matrix Communication
        CSRMatrix* communicate(const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double>& values);
        CSRMatrix* communicate_T(const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double>& values, 
                const int n_result_rows);
        CSRMatrix* communicate(const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices);
        CSRMatrix* communicate_T(const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const int n_result_rows);
        CSRMatrix* communicate(ParCSRMatrix* A)
        {
            return CommPkg::communicate(A);
        }
        CSRMatrix* communicate(CSRMatrix* A)
        {
            return CommPkg::communicate(A);
        }
        CSRMatrix* communicate_T(CSRMatrix* A)
        {
            return CommPkg::communicate_T(A);
        }

        // Vector Communication        
        aligned_vector<double>& communicate(ParVector& v)
        {
            return CommPkg::communicate(v);
        }

        void init_comm(ParVector& v)
        {
            CommPkg::init_comm(v);
        }

        // Helper Methods
        aligned_vector<double>& get_double_recv_buffer()
        {
            return recv_buffer;
        }
        aligned_vector<int>& get_int_recv_buffer()
        {
            return int_recv_buffer;
        }

        // Class Attributes
        ParComm* local_S_par_comm;
        ParComm* local_R_par_comm;
        ParComm* local_L_par_comm;
        ParComm* global_par_comm;
        aligned_vector<double> recv_buffer;
        aligned_vector<int> int_recv_buffer;
    };
}
#endif
