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
            comm_time = 0.0;
            comm_n = 0;
            comm_s = 0;
        }
        
        CommPkg(Topology* _topology)
        {
            topology = _topology;
            topology->num_shared++;
            comm_time = 0.0;
            comm_n = 0;
            comm_s = 0;
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


        static MPI_Datatype get_type(std::vector<int> buffer)
        {
            return MPI_INT;
        }
        static MPI_Datatype get_type(std::vector<double> buffer)
        {
            return MPI_DOUBLE;
        }


        // Matrix Communication
        virtual CSRMatrix* communicate(std::vector<int>& rowptr, 
                std::vector<int>& col_indices, std::vector<double>& values) = 0;
        virtual CSRMatrix* communicate_T(std::vector<int>& rowptr,
                std::vector<int>& col_indices, std::vector<double>& values, 
                int n_result_rows) = 0;
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
        std::vector<double>& communicate(ParVector& v);
        void init_comm(ParVector& v);

        // Standard Communication
        template<typename T>
        std::vector<T>& communicate(const std::vector<T>& values)
        {  
            return communicate(values.data());
        }
        template<typename T>
        void init_comm(const std::vector<T>& values)
        {
            init_comm(values.data());
        }
        template<typename T> void init_comm(const T* values);
        template<typename T> std::vector<T>& complete_comm();
        template<typename T> std::vector<T>& communicate(const T* values);
        virtual void init_double_comm(const double* values) = 0;
        virtual void init_int_comm(const int* values) = 0;
        virtual std::vector<double>& complete_double_comm() = 0;
        virtual std::vector<int>& complete_int_comm() = 0;

        // Transpose Communication
        template<typename T, typename U>
        void communicate_T(const std::vector<T>& values, std::vector<U>& result)
        {  
            communicate_T(values.data(), result);
        }
        template<typename T>
        void communicate_T(const std::vector<T>& values)
        {  
            communicate_T(values.data());
        }
        template<typename T>
        void init_comm_T(const std::vector<T>& values)
        {
            init_comm_T(values.data());
        }
        template<typename T> void init_comm_T(const T* values);
        template<typename T, typename U> void complete_comm_T(std::vector<U>& result);
        template<typename T> void complete_comm_T();
        template<typename T, typename U> void communicate_T(const T* values, std::vector<U>& result);
        template<typename T> void communicate_T(const T* values);
        virtual void init_double_comm_T(const double* values) = 0;
        virtual void init_int_comm_T(const int* values) = 0;
        virtual void complete_double_comm_T(std::vector<double>& result) = 0;
        virtual void complete_double_comm_T(std::vector<int>& result) = 0;
        virtual void complete_int_comm_T(std::vector<int>& result) = 0;
        virtual void complete_int_comm_T(std::vector<double>& result) = 0;
        virtual void complete_double_comm_T() = 0;
        virtual void complete_int_comm_T() = 0;

        // Conditional Communication
        template<typename T> std::vector<T>& conditional_comm(const std::vector<T>& values, 
                const std::vector<int>& send_compares, 
                const std::vector<int>& recv_compares, 
                std::function<bool(int)> compare_func = {})
        {  
            return conditional_comm(values.data(), send_compares.data(), recv_compares.data(), 
                    compare_func);
        }
        template<typename T> std::vector<T>& conditional_comm(const std::vector<T>& values, 
                const int* send_compares, 
                const int* recv_compares, 
                std::function<bool(int)> compare_func = {})
        {  
            return conditional_comm(values.data(), send_compares, recv_compares, 
                    compare_func);
        }
        template<typename T> std::vector<T>& conditional_comm(const T* values, 
                const std::vector<int>& send_compares, 
                const std::vector<int>& recv_compares, 
                std::function<bool(int)> compare_func = {})
        {  
            return conditional_comm(values, send_compares.data(), recv_compares.data(), 
                    compare_func);
        }
        template<typename T> std::vector<T>& conditional_comm(const T* values, 
                    const int* send_compares, 
                    const int* recv_compares,
                    std::function<bool(int)> compare_func = {});

        template<typename T, typename U> void conditional_comm_T(const std::vector<T>& values,
                std::vector<U>& result, 
                const std::vector<int>& send_compares, 
                const std::vector<int>& recv_compares, 
                std::function<bool(int)> compare_func = {},
                std::function<U(U, T)> result_func = {})
        {  
            return conditional_comm_T(values.data(), result, send_compares.data(), 
                    recv_compares.data(), compare_func, result_func);
        }
        template<typename T, typename U> void conditional_comm_T(const std::vector<T>& values,
                std::vector<U>& result, 
                const int* send_compares, 
                const int* recv_compares, 
                std::function<bool(int)> compare_func = {},
                std::function<U(U, T)> result_func = {})
        {  
            return conditional_comm_T(values.data(), result, send_compares, recv_compares, 
                    compare_func, result_func);
        }
        template<typename T, typename U> void conditional_comm_T(const T* values,
                std::vector<U>& result, 
                const std::vector<int>& send_compares, 
                const std::vector<int>& recv_compares, 
                std::function<bool(int)> compare_func = {},
                std::function<U(U, T)> result_func = {})
        {  
            return conditional_comm_T(values, result, send_compares.data(), 
                    recv_compares.data(), compare_func, result_func);
        }
        template<typename T, typename U> void conditional_comm_T(const T* values,
                    std::vector<U>& result, 
                    const int* send_compares, 
                    const int* recv_compares,
                    std::function<bool(int)> compare_func = {},
                    std::function<U(U, T)> result_func = {});

        virtual std::vector<double>& conditional_double_comm(const double* values,
                const int* send_compares,
                const int* recv_compares, 
                std::function<bool(int)> compare_func = {}) = 0;
        virtual std::vector<int>& conditional_int_comm(const int* values, 
                const int* send_compares,
                const int* recv_compares, 
                std::function<bool(int)> compare_func = {}) = 0;
        virtual void conditional_double_comm_T(const double* values, 
                std::vector<double>& result, 
                const int* send_compares,
                const int* recv_compares, 
                std::function<bool(int)> compare_func = {},
                std::function<double(double, double)> result_func = {}) = 0;
        virtual void conditional_int_comm_T(const int* values, 
                std::vector<int>& result,
                const int* send_compares,
                const int* recv_compares, 
                std::function<bool(int)> compare_func = {},
                std::function<int(int, int)> result_func = {}) = 0;
        virtual void conditional_int_comm_T(const int* values, 
                std::vector<double>& result,
                const int* send_compares,
                const int* recv_compares, 
                std::function<bool(int)> compare_func = {},
                std::function<double(double, int)> result_func = {}) = 0;

        // Helper methods
        template <typename T> std::vector<T>& get_recv_buffer();
        virtual std::vector<double>& get_double_recv_buffer() = 0;
        virtual std::vector<int>& get_int_recv_buffer() = 0;
        virtual double get_comm_time() = 0;
        virtual int get_comm_n() = 0;
        virtual int get_comm_s() = 0;

        // Class Variables
        Topology* topology;
        double comm_time;
        int comm_n;
        int comm_s;
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
    ***** send_procs : std::vector<index_t>
    *****    Distant processes messages are to be sent to
    ***** send_row_starts : std::vector<index_t>
    *****    Pointer to first position in send_row_indices
    *****    that a given process will send.
    ***** send_row_indices : std::vector<index_t> 
    *****    The indices of values that must be sent to each
    *****    process in send_procs
    ***** recv_procs : std::vector<index_t>
    *****    Distant processes messages are to be recvd from
    ***** recv_col_starts : std::vector<index_t>
    *****    Pointer to first column recvd from each process
    *****    in recv_procs
    ***** col_to_proc : std::vector<index_t>
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
        ***** off_proc_column_map : std::vector<index_t>&
        *****    Maps local off_proc columns indices to global
        ***** _key : int (optional)
        *****    Tag to be used in MPI Communication (default 9999)
        **************************************************************/
        ParComm(Partition* partition,
                const std::vector<int>& off_proc_column_map,
                int _key = 9999,
                MPI_Comm comm = MPI_COMM_WORLD) : CommPkg(partition)
        {
            mpi_comm = comm;
            init_par_comm(partition, off_proc_column_map, _key, comm);
        }

        ParComm(Partition* partition,
                const std::vector<int>& off_proc_column_map,
                const std::vector<int>& on_proc_column_map,
                int _key = 9999, 
                MPI_Comm comm = MPI_COMM_WORLD) : CommPkg(partition)
        {
            mpi_comm = comm;
            int idx;
            int ctr = 0;
            std::vector<int> part_col_to_new;

            init_par_comm(partition, off_proc_column_map, _key, comm);
            
            if (partition->local_num_cols)
            {
                part_col_to_new.resize(partition->local_num_cols, -1);
            }
            for (std::vector<int>::const_iterator it = on_proc_column_map.begin();
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
                const std::vector<int>& off_proc_column_map,
                int _key, MPI_Comm comm)
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

            std::vector<int> off_proc_col_to_proc(off_proc_num_cols);
            std::vector<int> tmp_send_buffer;

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

        ParComm(ParComm* comm, const std::vector<int>& off_proc_col_to_new)
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

            comm->communicate_T(off_proc_col_to_new);

            recv_data->size_msgs = 0;
            if (comm->recv_data->indices.size())
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
            else
            {
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

        ParComm(ParComm* comm, const std::vector<int>& on_proc_col_to_new,
                const std::vector<int>& off_proc_col_to_new) 
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

            comm->communicate_T(off_proc_col_to_new);

            recv_data->size_msgs = 0;
            if (comm->recv_data->indices.size())
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
            else
            {
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
                    idx = comm->send_data->indices[j];
                    new_idx = on_proc_col_to_new[idx];
                    if (new_idx != -1 && comm->send_data->int_buffer[j] != -1)
                    {
                        comm_proc = true;
                        send_data->indices.push_back(new_idx);
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

        // Standard Communication
        void init_double_comm(const double* values)
        {
            initialize(values);
        }
        void init_int_comm(const int* values)
        {
            initialize(values);
        }
        std::vector<double>& complete_double_comm()
        {
            return complete<double>();
        }
        std::vector<int>& complete_int_comm()
        {
            return complete<int>();
        }
        template<typename T>
        std::vector<T>& communicate(const std::vector<T>& values)
        {
            return CommPkg::communicate(values.data());
        }
        template<typename T>
        std::vector<T>& communicate(const T* values)
        {
            return CommPkg::communicate(values);
        }

        template<typename T>
        void initialize(const T* values)
        {
            comm_time -= MPI_Wtime();
            int start, end;
            int proc;

            std::vector<T>& sendbuf = send_data->get_buffer<T>();
            std::vector<T>& recvbuf = recv_data->get_buffer<T>();
            MPI_Datatype type = get_type(sendbuf);
            send_data->vector_data.num_msgs += send_data->num_msgs;
            send_data->vector_data.size_msgs += send_data->size_msgs;

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
            comm_time += MPI_Wtime();
            comm_n += send_data->num_msgs;
            comm_s += (send_data->size_msgs * sizeof(T));
        }

        template<typename T>
        std::vector<T>& complete()
        {
            comm_time -= MPI_Wtime();
            if (send_data->num_msgs)
            {
                MPI_Waitall(send_data->num_msgs, send_data->requests.data(), MPI_STATUS_IGNORE);
            }

            if (recv_data->num_msgs)
            {
                MPI_Waitall(recv_data->num_msgs, recv_data->requests.data(), MPI_STATUS_IGNORE);
            }
            comm_time += MPI_Wtime();

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
        void complete_double_comm_T(std::vector<double>& result)
        {
            complete_T<double>(result);
        }
        void complete_double_comm_T(std::vector<int>& result)
        {
            complete_T<double>(result);
        }
        void complete_int_comm_T(std::vector<double>& result)
        {
            complete_T<int>(result);
        }
        void complete_int_comm_T(std::vector<int>& result)
        {
            complete_T<int>(result);
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
        void communicate_T(const std::vector<T>& values, std::vector<U>& result)
        {
            CommPkg::communicate_T(values.data(), result);
        }
        template<typename T, typename U>
        void communicate_T(const T* values, std::vector<U>& result)
        {
            CommPkg::communicate_T(values, result);
        }
        template<typename T>
        void communicate_T(const std::vector<T>& values)
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
            comm_time -= MPI_Wtime();
            int start, end;
            int proc, idx;
            std::vector<T>& sendbuf = send_data->get_buffer<T>();
            std::vector<T>& recvbuf = recv_data->get_buffer<T>();
            MPI_Datatype type = get_type(sendbuf);
            recv_data->vector_data.num_msgs += recv_data->num_msgs;
            recv_data->vector_data.size_msgs += recv_data->size_msgs;

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
            comm_time += MPI_Wtime();
            comm_n += recv_data->num_msgs;
            comm_s += (recv_data->size_msgs * sizeof(T));
        }

        template<typename T, typename U>
        void complete_T(std::vector<U>& result)
        {
            complete_T<T>();

            int idx;
            std::vector<T>& sendbuf = send_data->get_buffer<T>();
            for (int i = 0; i < send_data->size_msgs; i++)
            {
                idx = send_data->indices[i];
                result[idx] += sendbuf[i];
            }
        }

        template<typename T>
        void complete_T()
        {
            comm_time -= MPI_Wtime();
            if (send_data->num_msgs)
            {
                MPI_Waitall(send_data->num_msgs, send_data->requests.data(), MPI_STATUSES_IGNORE);
            }

            if (recv_data->num_msgs)
            {
                MPI_Waitall(recv_data->num_msgs, recv_data->requests.data(), MPI_STATUSES_IGNORE);
            }
            comm_time += MPI_Wtime();
        }


        // Matrix Communication
        CSRMatrix* communicate(std::vector<int>& rowptr, std::vector<int>& col_indices,
                std::vector<double>& values);
        CSRMatrix* communicate_T(std::vector<int>& rowptr, std::vector<int>& col_indices,
                std::vector<double>& values, int n_result_rows);
        CSRMatrix* communication_helper(std::vector<int>& rowptr, 
                std::vector<int>& col_indices, std::vector<double>& values,
                CommData* send_comm, CommData* recv_comm);
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
        std::vector<double>& communicate(ParVector& v)
        {
            return CommPkg::communicate(v);
        }
        void init_comm(ParVector& v)
        {
            CommPkg::init_comm(v);
        }


        // Conditional Communication
        std::vector<double>& conditional_double_comm(const double* values, 
                const int* send_compares,
                const int* recv_compares, 
                std::function<bool(int)> compare_func = {})
        {
            return conditional_communication<double>(values, send_compares, 
                    recv_compares, compare_func);
        
        }
        std::vector<int>& conditional_int_comm(const int* values,
                const int* send_compares,
                const int* recv_compares, 
                std::function<bool(int)> compare_func = {})
        {
            return conditional_communication<int>(values, send_compares, 
                    recv_compares, compare_func);
        }

        void conditional_double_comm_T(const double* values, 
                std::vector<double>& result,
                const int* send_compares,
                const int* recv_compares, 
                std::function<bool(int)> compare_func = {},
                std::function<double(double, double)> result_func = {})
        {
            conditional_communication_T<double, double>(values, result, 
                    send_compares, recv_compares, compare_func, result_func);
        }

        void conditional_int_comm_T(const int* values, 
                std::vector<int>& result,
                const int* send_compares,
                const int* recv_compares, 
                std::function<bool(int)> compare_func = {},
                std::function<int(int, int)> result_func = {})
        {
            conditional_communication_T<int, int>(values, result, send_compares, 
                    recv_compares, compare_func, result_func);
        }

        void conditional_int_comm_T(const int* values, 
                std::vector<double>& result,
                const int* send_compares,
                const int* recv_compares, 
                std::function<bool(int)> compare_func = {},
                std::function<double(double, int)> result_func = {})
        {
            conditional_communication_T<int, double>(values, result, 
                    send_compares, recv_compares, compare_func, result_func);
        }

        template<typename T>
        std::vector<T>& conditional_communication(const T* values, 
                const int* send_compares,
                const int* recv_compares,
                std::function<bool(int)> compare_func = {})
        {
            comm_time -= MPI_Wtime();
            if (!compare_func) return communicate(values);

            int proc, start, end;
            int idx, size;
            int ctr, prev_ctr;
            int n_sends, n_recvs;

            std::vector<T>& sendbuf = send_data->get_buffer<T>();
            std::vector<T>& recvbuf = recv_data->get_buffer<T>();
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
                    if (compare_func(send_compares[idx]))
                    {
                        sendbuf[ctr++] = values[idx];
                    }
                }
                size = ctr - prev_ctr;
                if (size)
                {
                    MPI_Issend(&(sendbuf[prev_ctr]), size, type, 
                            proc, key, mpi_comm, &(send_data->requests[n_sends++]));
                    prev_ctr = ctr;
                }
            }
            

            n_recvs = 0;
            ctr = 0;
            prev_ctr = 0;
            if (recv_data->indices.size())
            {
                for (int i = 0; i < recv_data->num_msgs; i++)
                {
                    proc = recv_data->procs[i];
                    start = recv_data->indptr[i];
                    end = recv_data->indptr[i+1];
                    for (int j = start; j < end; j++)
                    {
                        idx = recv_data->indices[j];

                        if (compare_func(recv_compares[idx]))
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

                        if (compare_func(recv_compares[idx]))
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
            if (recv_data->indices.size())
            {
                for (int i = recv_data->size_msgs - 1; i >= 0; i--)
                {
                    idx = recv_data->indices[i];

                    if (compare_func(recv_compares[idx]))
                    {
                        recvbuf[idx] = recvbuf[ctr--];
                    }
                    else
                    {
                        recvbuf[idx] = 0.0;
                    }
                }
            }
            else
            {
                for (int i = recv_data->size_msgs - 1; i >= 0; i--)
                {
                    idx = i;

                    if (compare_func(recv_compares[idx]))
                    {
                        recvbuf[idx] = recvbuf[ctr--];
                    }
                    else
                    {
                        recvbuf[idx] = 0.0;
                    }
                }
            }
            return recvbuf;
            comm_time += MPI_Wtime();
        }

        template<typename T, typename U>
        void conditional_communication_T(const T* values,
                std::vector<U>& result, 
                const int* send_compares,
                const int* recv_compares,
                std::function<bool(int)> compare_func = {},
                std::function<U(U, T)> result_func = {})
        {
            comm_time -= MPI_Wtime();
            if (!compare_func)
            {
                communicate_T(values, result);
                return;
            }

            int proc, start, end;
            int idx, size;
            int ctr, prev_ctr;
            int n_sends, n_recvs;

            std::vector<T>& sendbuf = send_data->get_buffer<T>();
            std::vector<T>& recvbuf = recv_data->get_buffer<T>();
            MPI_Datatype type = get_type(sendbuf);

            n_sends = 0;
            ctr = 0;
            prev_ctr = 0;
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
                        if (compare_func(recv_compares[idx]))
                        {
                            idx_start = recv_data->indptr_T[j];
                            idx_end = recv_data->indptr_T[j+1];
                            val = 0;
                            for (int k = idx_start; k < idx_end; k++)
                            {
                                val += values[recv_data->indices[k]];
                            }
                            recvbuf[ctr++] = val;
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

                        if (compare_func(recv_compares[idx]))
                        {
                            recvbuf[ctr++] = values[idx];
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

                        if (compare_func(recv_compares[idx]))
                        {
                            recvbuf[ctr++] = values[idx];
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
                    if (compare_func(send_compares[idx]))
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

            if (result_func)
            {
                ctr = 0;
                for (int i = 0; i < send_data->size_msgs; i++)
                {
                    idx = send_data->indices[i];
                    if (compare_func(send_compares[idx]))
                    {
                        result[idx] = result_func(result[idx], sendbuf[ctr++]);
                    }
                }
            }
            comm_time += MPI_Wtime();
        }

        double get_comm_time()
        {
            return comm_time;
        }

        int get_comm_n()
        {
            return comm_n;
        }

        int get_comm_s()
        {
            return comm_s;
        }

        // Helper Methods
        std::vector<double>& get_double_recv_buffer()
        {
            return recv_data->buffer;
        }
        std::vector<int>& get_int_recv_buffer()
        {
            return recv_data->int_buffer;
        }
        std::vector<double>& get_double_send_buffer()
        {
            return send_data->buffer;
        }
        std::vector<int>& get_int_send_buffer()
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
        ***** off_proc_column_map : std::vector<int>&
        *****    Maps local off_proc columns indices to global
        ***** global_num_cols : int
        *****    Number of global columns in matrix
        ***** local_num_cols : int
        *****    Number of columns local to rank
        **************************************************************/
        TAPComm(Partition* partition, 
                const std::vector<int>& off_proc_column_map,
                MPI_Comm comm = MPI_COMM_WORLD) : CommPkg(partition)
        {
            init_tap_comm(partition, off_proc_column_map, comm);
        }

        TAPComm(Partition* partition,
                const std::vector<int>& off_proc_column_map,
                const std::vector<int>& on_proc_column_map,
                MPI_Comm comm = MPI_COMM_WORLD) : CommPkg(partition)
        {
            init_tap_comm(partition, off_proc_column_map, comm);

            std::vector<int> on_proc_to_new;
            int on_proc_num_cols = on_proc_column_map.size();
            if (partition->local_num_cols)
            {
                on_proc_to_new.resize(partition->local_num_cols);
                for (int i = 0; i < on_proc_num_cols; i++)
                {
                    on_proc_to_new[on_proc_column_map[i] - partition->first_local_col] = i;
                }
            }
            
            for (std::vector<int>::iterator it = local_S_par_comm->send_data->indices.begin();
                    it != local_S_par_comm->send_data->indices.end(); ++it)
            {
                *it = on_proc_to_new[*it];
            }

            for (std::vector<int>::iterator it = local_L_par_comm->send_data->indices.begin();
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
            global_par_comm = new ParComm(tap_comm->global_par_comm);
            local_S_par_comm = new ParComm(tap_comm->local_S_par_comm);
            local_R_par_comm = new ParComm(tap_comm->local_R_par_comm);
            local_L_par_comm = new ParComm(tap_comm->local_L_par_comm);

            int recv_size = tap_comm->recv_buffer.size();
            if (recv_size)
            {
                recv_buffer.resize(recv_size);
                int_recv_buffer.resize(recv_size);
            }
        }

        TAPComm(TAPComm* tap_comm, const std::vector<int>& on_proc_col_to_new,
                const std::vector<int>& off_proc_col_to_new, bool communicate = false) 
            : CommPkg(tap_comm->topology)
        {
            bool comm_proc;
            int proc, start, end;
            int idx, new_idx;
            
            int new_idx_L;
            int new_idx_R;

            local_L_par_comm = new ParComm(tap_comm->topology, 
                    tap_comm->local_L_par_comm->key, tap_comm->local_L_par_comm->mpi_comm);
            local_S_par_comm = new ParComm(tap_comm->topology,
                    tap_comm->local_S_par_comm->key, tap_comm->local_S_par_comm->mpi_comm);
            local_R_par_comm = new ParComm(tap_comm->topology,
                    tap_comm->local_R_par_comm->key, tap_comm->local_R_par_comm->mpi_comm);
            global_par_comm = new ParComm(tap_comm->topology,
                    tap_comm->global_par_comm->key, tap_comm->global_par_comm->mpi_comm);
            
            // Communicate the col_to_new lists to other local procs
            tap_comm->local_S_par_comm->communicate(on_proc_col_to_new);
            tap_comm->local_R_par_comm->communicate_T(off_proc_col_to_new);

            // Form col_to_new for S_recv, R_send, and global_recv
            std::vector<int>& S_recv_col_to_new 
                = tap_comm->local_S_par_comm->recv_data->int_buffer;
            std::vector<int>& R_send_col_to_new = 
                tap_comm->local_R_par_comm->send_data->int_buffer;
            std::vector<int> global_recv_col_to_new;
            if (tap_comm->global_par_comm->recv_data->size_msgs)
            {
                global_recv_col_to_new.resize(tap_comm->global_par_comm->recv_data->size_msgs);
            }
            for (int i = 0; i < tap_comm->local_R_par_comm->send_data->size_msgs; i++)
            {
                idx = tap_comm->local_R_par_comm->send_data->indices[i];
                global_recv_col_to_new[idx] = R_send_col_to_new[i];
            }

            // Update local_L_par_comm
            for (int i = 0; i < tap_comm->local_L_par_comm->send_data->num_msgs; i++)
            {
                comm_proc = false;
                start = tap_comm->local_L_par_comm->send_data->indptr[i];
                end = tap_comm->local_L_par_comm->send_data->indptr[i+1];
                for (int j = start; j < end; j++)
                {
                    idx = tap_comm->local_L_par_comm->send_data->indices[j];
                    new_idx = on_proc_col_to_new[idx];
                    if (new_idx != -1)
                    {
                        comm_proc = true;
                        local_L_par_comm->send_data->indices.push_back(new_idx);
                    }
                }
                if (comm_proc)
                {
                    proc = tap_comm->local_L_par_comm->send_data->procs[i];
                    local_L_par_comm->send_data->procs.push_back(proc);
                    local_L_par_comm->send_data->indptr.push_back(
                            local_L_par_comm->send_data->indices.size());
                }
            }
            local_L_par_comm->send_data->num_msgs = local_L_par_comm->send_data->procs.size();
            local_L_par_comm->send_data->size_msgs = local_L_par_comm->send_data->indices.size();
            local_L_par_comm->send_data->finalize();

            for (int i = 0; i < tap_comm->local_L_par_comm->recv_data->num_msgs; i++)
            {
                comm_proc = false;
                start = tap_comm->local_L_par_comm->recv_data->indptr[i];
                end = tap_comm->local_L_par_comm->recv_data->indptr[i+1];
                for (int j = start; j < end; j++)
                {
                    idx = tap_comm->local_L_par_comm->recv_data->indices[j];
                    new_idx = off_proc_col_to_new[idx];
                    if (new_idx != -1)
                    {
                        comm_proc = true;
                        local_L_par_comm->recv_data->indices.push_back(new_idx);
                    }
                }
                if (comm_proc)
                {
                    proc = tap_comm->local_L_par_comm->recv_data->procs[i];
                    local_L_par_comm->recv_data->procs.push_back(proc);
                    local_L_par_comm->recv_data->indptr.push_back(
                            local_L_par_comm->recv_data->indices.size());
                }
            }
            local_L_par_comm->recv_data->num_msgs = local_L_par_comm->recv_data->procs.size();
            local_L_par_comm->recv_data->size_msgs = local_L_par_comm->recv_data->indices.size();
            local_L_par_comm->recv_data->finalize();



            // Update local_S_par_comm
            for (int i = 0; i < tap_comm->local_S_par_comm->send_data->num_msgs; i++)
            {
                comm_proc = false;
                start = tap_comm->local_S_par_comm->send_data->indptr[i];
                end = tap_comm->local_S_par_comm->send_data->indptr[i+1];
                for (int j = start; j < end; j++)
                {
                    idx = tap_comm->local_S_par_comm->send_data->indices[j];
                    new_idx = on_proc_col_to_new[idx];
                    if (new_idx != -1)
                    {
                        comm_proc = true;
                        local_S_par_comm->send_data->indices.push_back(new_idx);
                    }
                }
                if (comm_proc)
                {
                    proc = tap_comm->local_S_par_comm->send_data->procs[i];
                    local_S_par_comm->send_data->procs.push_back(proc);
                    local_S_par_comm->send_data->indptr.push_back(
                            local_S_par_comm->send_data->indices.size());
                }
            }
            local_S_par_comm->send_data->num_msgs = local_S_par_comm->send_data->procs.size();
            local_S_par_comm->send_data->size_msgs = local_S_par_comm->send_data->indices.size();
            local_S_par_comm->send_data->finalize();

            local_S_par_comm->recv_data->size_msgs = 0;
            for (int i = 0; i < tap_comm->local_S_par_comm->recv_data->num_msgs; i++)
            {
                comm_proc = false;
                start = tap_comm->local_S_par_comm->recv_data->indptr[i];
                end = tap_comm->local_S_par_comm->recv_data->indptr[i+1];
                for (int j = start; j < end; j++)
                {
                    new_idx = S_recv_col_to_new[j];
                    if (new_idx != -1)
                    {
                        comm_proc = true;
                        local_S_par_comm->recv_data->size_msgs++;
                    }
                }
                if (comm_proc)
                {
                    proc = tap_comm->local_S_par_comm->recv_data->procs[i];
                    local_S_par_comm->recv_data->procs.push_back(proc);
                    local_S_par_comm->recv_data->indptr.push_back(
                            local_S_par_comm->recv_data->size_msgs);
                }
            }
            local_S_par_comm->recv_data->num_msgs = local_S_par_comm->recv_data->procs.size();
            local_S_par_comm->recv_data->finalize();


            // Update local_R_par_comm
            for (int i = 0; i < tap_comm->local_R_par_comm->recv_data->num_msgs; i++)
            {
                comm_proc = false;
                start = tap_comm->local_R_par_comm->recv_data->indptr[i];
                end = tap_comm->local_R_par_comm->recv_data->indptr[i+1];
                for (int j = start; j < end; j++)
                {
                    idx = tap_comm->local_R_par_comm->recv_data->indices[j];
                    new_idx = off_proc_col_to_new[idx];
                    if (new_idx != -1)
                    {
                        comm_proc = true;
                        local_R_par_comm->recv_data->indices.push_back(new_idx);
                    }
                }
                if (comm_proc)
                {
                    proc = tap_comm->local_R_par_comm->recv_data->procs[i];
                    local_R_par_comm->recv_data->procs.push_back(proc);
                    local_R_par_comm->recv_data->indptr.push_back(
                            local_R_par_comm->recv_data->indices.size());
                }
            }
            local_R_par_comm->recv_data->num_msgs = local_R_par_comm->recv_data->procs.size();
            local_R_par_comm->recv_data->size_msgs = local_R_par_comm->recv_data->indices.size();
            local_R_par_comm->recv_data->finalize();

            for (int i = 0; i < tap_comm->local_R_par_comm->send_data->num_msgs; i++)
            {
                comm_proc = false;
                start = tap_comm->local_R_par_comm->send_data->indptr[i];
                end = tap_comm->local_R_par_comm->send_data->indptr[i+1];
                for (int j = start; j < end; j++)
                {
                    idx = tap_comm->local_R_par_comm->send_data->indices[j];
                    new_idx = R_send_col_to_new[idx];
                    if (new_idx != -1)
                    {
                        comm_proc = true;
                        local_R_par_comm->send_data->indices.push_back(new_idx);
                    }
                }
                if (comm_proc)
                {
                    proc = tap_comm->local_R_par_comm->send_data->procs[i];
                    local_R_par_comm->send_data->procs.push_back(proc);
                    local_R_par_comm->send_data->indptr.push_back(
                            local_R_par_comm->send_data->indices.size());
                }
            }
            local_R_par_comm->send_data->num_msgs = local_R_par_comm->send_data->procs.size();
            local_R_par_comm->send_data->size_msgs = local_R_par_comm->send_data->indices.size();
            local_R_par_comm->send_data->finalize();



            // Update global par comm
            for (int i = 0; i < tap_comm->global_par_comm->send_data->num_msgs; i++)
            {
                comm_proc = false;
                start = tap_comm->global_par_comm->send_data->indptr[i];
                end = tap_comm->global_par_comm->send_data->indptr[i+1];
                for (int j = start; j < end; j++)
                {
                    idx = tap_comm->global_par_comm->send_data->indices[j];
                    new_idx = S_recv_col_to_new[idx];
                    if (new_idx != -1)
                    {
                        comm_proc = true;
                        global_par_comm->send_data->indices.push_back(new_idx);
                    }
                }
                if (comm_proc)
                {
                    proc = tap_comm->global_par_comm->send_data->procs[i];
                    global_par_comm->send_data->procs.push_back(proc);
                    global_par_comm->send_data->indptr.push_back(
                            global_par_comm->send_data->indices.size());
                }
            }
            global_par_comm->send_data->num_msgs = global_par_comm->send_data->procs.size();
            global_par_comm->send_data->size_msgs = global_par_comm->send_data->indices.size();
            global_par_comm->send_data->finalize();


            global_par_comm->recv_data->size_msgs = 0;
            for (int i = 0; i < tap_comm->global_par_comm->recv_data->num_msgs; i++)
            {
                comm_proc = false;
                start = tap_comm->global_par_comm->recv_data->indptr[i];
                end = tap_comm->global_par_comm->recv_data->indptr[i+1];
                for (int j = start; j < end; j++)
                {
                    new_idx = global_recv_col_to_new[j];
                    if (new_idx != -1)
                    {
                        comm_proc = true;
                        global_par_comm->recv_data->size_msgs++;
                    }
                }
                if (comm_proc)
                {
                    proc = tap_comm->global_par_comm->recv_data->procs[i];
                    global_par_comm->recv_data->procs.push_back(proc);
                    global_par_comm->recv_data->indptr.push_back(
                            global_par_comm->recv_data->size_msgs);
                }
            }
            global_par_comm->recv_data->num_msgs = global_par_comm->recv_data->procs.size();
            global_par_comm->recv_data->finalize();

            
            int recv_size = local_R_par_comm->recv_data->size_msgs +
                local_L_par_comm->recv_data->size_msgs;
            if (recv_size)
            {
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
            delete local_S_par_comm;
            delete local_R_par_comm;
            delete local_L_par_comm;
        }

        void init_tap_comm(Partition* partition,
                const std::vector<int>& off_proc_column_map,
                MPI_Comm comm)
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
            std::vector<int> off_proc_col_to_proc;
            std::vector<int> on_node_column_map;
            std::vector<int> on_node_col_to_proc;
            std::vector<int> off_node_column_map;
            std::vector<int> off_node_col_to_node;
            std::vector<int> on_node_to_off_proc;
            std::vector<int> off_node_to_off_proc;
            std::vector<int> recv_nodes;
            std::vector<int> orig_procs;
            std::vector<int> node_to_local_proc;

            // Find process on which vector value associated with each column is
            // stored
            partition->form_col_to_proc(off_proc_column_map, off_proc_col_to_proc);

            // Partition off_proc cols into on_node and off_node
            split_off_proc_cols(off_proc_column_map, off_proc_col_to_proc,
                   on_node_column_map, on_node_col_to_proc, on_node_to_off_proc,
                   off_node_column_map, off_node_col_to_node, off_node_to_off_proc);

            // Gather all nodes with which any local process must communication
            form_local_R_par_comm(off_node_column_map, off_node_col_to_node, 
                    orig_procs);

            // Find global processes with which rank communications
            form_global_par_comm(orig_procs);

            // Form local_S_par_comm: initial distribution of values among local
            // processes, before inter-node communication
            form_local_S_par_comm(orig_procs);

            // Adjust send indices (currently global vector indices) to be index 
            // of global vector value from previous recv
            adjust_send_indices(partition->first_local_col);

            // Form local_L_par_comm: fully local communication (origin and
            // destination processes both local to node)
            form_local_L_par_comm(on_node_column_map, on_node_col_to_proc,
                    partition->first_local_col);

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

        // Helper methods for forming TAPComm:
        void split_off_proc_cols(const std::vector<int>& off_proc_column_map,
                const std::vector<int>& off_proc_col_to_proc,
                std::vector<int>& on_node_column_map,
                std::vector<int>& on_node_col_to_proc,
                std::vector<int>& on_node_to_off_proc,
                std::vector<int>& off_node_column_map,
                std::vector<int>& off_node_col_to_node,
                std::vector<int>& off_node_to_off_proc);
        void form_local_R_par_comm(const std::vector<int>& off_node_column_map,
                const std::vector<int>& off_node_col_to_node,
                std::vector<int>& orig_procs);
        void form_global_par_comm(std::vector<int>& orig_procs);
        void form_local_S_par_comm(std::vector<int>& orig_procs);
        void adjust_send_indices(const int first_local_col);
        void form_local_L_par_comm(const std::vector<int>& on_node_column_map,
                const std::vector<int>& on_node_col_to_proc,
                const int first_local_col);

        // Class Methods
        void init_double_comm(const double* values)
        {
            initialize(values);
        }
        void init_int_comm(const int* values)
        {
            initialize(values);
        }
        std::vector<double>& complete_double_comm()
        {
            return complete<double>();
        }
        std::vector<int>& complete_int_comm()
        {
            return complete<int>();
        }
        
        template<typename T>
        std::vector<T>& communicate(const std::vector<T>& values)
        {
            return CommPkg::communicate<T>(values.data());
        }
        template<typename T>
        std::vector<T>& communicate(const T* values)
        {
            return CommPkg::communicate<T>(values);
        }

        template<typename T>
        void initialize(const T* values)
        {
            // Messages with origin and final destination on node
            local_L_par_comm->communicate<T>(values);

            // Initial redistribution among node
            std::vector<T>& S_vals = 
                local_S_par_comm->communicate<T>(values);

            // Begin inter-node communication 
            global_par_comm->initialize(S_vals.data());
        }

        template<typename T>
        std::vector<T>& complete()
        {
            // Complete inter-node communication
            std::vector<T>& G_vals = global_par_comm->complete<T>();

            // Redistributing recvd inter-node values
            local_R_par_comm->communicate<T>(G_vals.data());

            std::vector<T>& recvbuf = get_recv_buffer<T>();
            std::vector<T>& R_recvbuf = local_R_par_comm->recv_data->get_buffer<T>();
            std::vector<T>& L_recvbuf = local_L_par_comm->recv_data->get_buffer<T>();

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
        void complete_double_comm_T(std::vector<double>& result)
        {
            complete_T<double>(result);
        }        
        void complete_double_comm_T(std::vector<int>& result)
        {
            complete_T<double>(result);
        }
        void complete_int_comm_T(std::vector<double>& result)
        {
            complete_T<int>(result);
        }
        void complete_int_comm_T(std::vector<int>& result)
        {
            complete_T<int>(result);
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
        void communicate_T(const std::vector<T>& values, std::vector<U>& result)
        {
            CommPkg::communicate_T(values.data(), result);
        }
        template<typename T, typename U>
        void communicate_T(const T* values, std::vector<U>& result)
        {
            CommPkg::communicate_T(values, result);
        }
        template<typename T>
        void communicate_T(const std::vector<T>& values)
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
            std::vector<T>& R_sendbuf = local_R_par_comm->send_data->get_buffer<T>();
            global_par_comm->init_comm_T(R_sendbuf);
        }

        template<typename T, typename U>
        void complete_T(std::vector<U>& result)
        {
            complete_T<T>();

            int idx;
            std::vector<T>& S_sendbuf = local_S_par_comm->send_data->get_buffer<T>();
            std::vector<T>& L_sendbuf = local_L_par_comm->send_data->get_buffer<T>();

            for (int i = 0; i < local_L_par_comm->send_data->size_msgs; i++)
            {
                idx = local_L_par_comm->send_data->indices[i];
                result[idx] += L_sendbuf[i];
            }
            for (int i = 0; i < local_S_par_comm->send_data->size_msgs; i++)
            {
                idx = local_S_par_comm->send_data->indices[i];
                result[idx] += S_sendbuf[i];
            } 
        }
        template<typename T>
        void complete_T()
        {
            // Complete inter-node communication
            global_par_comm->complete_comm_T<T>();

            int idx;
            std::vector<T>& G_sendbuf = global_par_comm->send_data->get_buffer<T>();
            local_S_par_comm->communicate_T(G_sendbuf);
        }


        // Matrix Communication
        CSRMatrix* communicate(std::vector<int>& rowptr, std::vector<int>& col_indices,
                std::vector<double>& values);
        CSRMatrix* communicate_T(std::vector<int>& rowptr, std::vector<int>& col_indices, 
                std::vector<double>& values, int n_result_rows);
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
        std::vector<double>& communicate(ParVector& v)
        {
            return CommPkg::communicate(v);
        }

        void init_comm(ParVector& v)
        {
            CommPkg::init_comm(v);
        }


        // Conditional Communication
        std::vector<double>& conditional_double_comm(const double* values, 
                const int* send_compares,
                const int* recv_compares, 
                std::function<bool(int)> compare_func = {})
        {
            return conditional_communication<double>(values, send_compares, 
                    recv_compares, compare_func);
        
        }
        std::vector<int>& conditional_int_comm(const int* values,
                const int* send_compares,
                const int* recv_compares, 
                std::function<bool(int)> compare_func = {})
        {
            return conditional_communication<int>(values, send_compares, 
                    recv_compares, compare_func);
        }

        void conditional_double_comm_T(const double* values, 
                std::vector<double>& result,
                const int* send_compares,
                const int* recv_compares, 
                std::function<bool(int)> compare_func = {},
                std::function<double(double, double)> result_func = {})
        {
            conditional_communication_T<double, double>(values, result, 
                    send_compares, recv_compares, compare_func, result_func);
        }
        void conditional_int_comm_T(const int* values, 
                std::vector<int>& result,
                const int* send_compares,
                const int* recv_compares, 
                std::function<bool(int)> compare_func = {},
                std::function<int(int, int)> result_func = {})                
        {
            conditional_communication_T<int, int>(values, result, 
                    send_compares, recv_compares, compare_func, result_func);
        }
        void conditional_int_comm_T(const int* values, 
                std::vector<double>& result,
                const int* send_compares,
                const int* recv_compares, 
                std::function<bool(int)> compare_func = {},
                std::function<double(double, int)> result_func = {})                
        {
            conditional_communication_T<int, double>(values, result, 
                    send_compares, recv_compares, compare_func, result_func);
        }

        template<typename T>
        std::vector<T>& conditional_communication(const T* values, 
                const int* send_compares,
                const int* recv_compares,
                std::function<bool(int)> compare_func = {})
        {
            int start, end;
            int idx, new_idx;
            int proc, size;
            int n_sends, n_recvs;
            int ctr;
            bool send_msg;

            std::vector<int> global_send_compares;
            std::vector<int> global_recv_compares;
            std::vector<T> L_recvbuf;
            std::vector<T> S_recvbuf;
            std::vector<T> R_recvbuf;
            std::vector<T> G_recvbuf;

            // Local communication... send states and off proc states
            local_S_par_comm->communicate(send_compares);
            std::copy(local_S_par_comm->recv_data->int_buffer.begin(),
                    local_S_par_comm->recv_data->int_buffer.end(),
                    std::back_inserter(global_send_compares));
            local_R_par_comm->communicate_T(recv_compares);
            if (global_par_comm->recv_data->size_msgs)
            {
                global_recv_compares.resize(global_par_comm->recv_data->size_msgs);
            }
            for (int i = 0; i < local_R_par_comm->send_data->size_msgs; i++)
            {
                idx = local_R_par_comm->send_data->indices[i];
                global_recv_compares[idx] = local_R_par_comm->send_data->int_buffer[i];
            }

            // Local communication... can send / recv everything
            S_recvbuf = local_S_par_comm->communicate(values);

            // Global communication... only send if compare yields true
            G_recvbuf = global_par_comm->conditional_comm(S_recvbuf,
                    global_send_compares,  
                    global_recv_compares, 
                    compare_func);
            //G_recvbuf = global_par_comm->communicate(S_recvbuf, comm);

            // local communication ... send as normal
            R_recvbuf = local_R_par_comm->communicate(G_recvbuf);
            L_recvbuf = local_L_par_comm->communicate(values);

            // Add values to recv_buffer
            std::vector<T>& recvbuf = get_recv_buffer<T>();

            for (int i = 0; i < L_recvbuf.size(); i++)
            {
                idx = local_L_par_comm->recv_data->indices[i];
                if (compare_func(recv_compares[idx]))
                {
                    recvbuf[idx] = L_recvbuf[i];
                }
                else
                {
                    recvbuf[idx] = 0.0;
                }
            }
            for (int i = 0; i < local_R_par_comm->recv_data->size_msgs; i++)
            {
                idx = local_R_par_comm->recv_data->indices[i];
                if (compare_func(recv_compares[idx]))
                {
                    recvbuf[idx] = R_recvbuf[i];
                }
                else
                {
                    recvbuf[idx] = 0.0;
                }
            }

            return recvbuf; 
        }

        template<typename T, typename U>
        void conditional_communication_T(const T* values,
                std::vector<U>& result, 
                const int* send_compares,
                const int* recv_compares,
                std::function<bool(int)> compare_func = {},
                std::function<U(U, T)> result_func = {})
        {
            int idx, new_idx;

            std::vector<T> L_sendbuf;
            std::vector<T> R_sendbuf;
            std::vector<T> G_sendbuf;
            std::vector<T> S_sendbuf;

            std::vector<T> G_recvbuf;
            std::vector<T> S_recvbuf;

            std::vector<int> global_send_compares;
            std::vector<int> global_recv_compares;

            // Local communication... send states and off proc states
            local_S_par_comm->communicate(send_compares);
            std::copy(local_S_par_comm->recv_data->int_buffer.begin(),
                    local_S_par_comm->recv_data->int_buffer.end(),
                    std::back_inserter(global_send_compares));
            local_R_par_comm->communicate_T(recv_compares);
            if (global_par_comm->recv_data->size_msgs)
            {
                global_recv_compares.resize(global_par_comm->recv_data->size_msgs);
            }
            for (int i = 0; i < local_R_par_comm->send_data->size_msgs; i++)
            {
                idx = local_R_par_comm->send_data->indices[i];
                global_recv_compares[idx] = local_R_par_comm->send_data->int_buffer[i];
            }

            // Initial redistribution among node
            local_R_par_comm->communicate_T(values);
            R_sendbuf = local_R_par_comm->send_data->get_buffer<T>();

            // Begin inter-node communication 
            G_recvbuf = global_par_comm->recv_data->get_buffer<T>();
            std::fill(G_recvbuf.begin(), G_recvbuf.end(), 0);
            for (int i = 0; i < local_R_par_comm->send_data->size_msgs; i++)
            {
                idx = local_R_par_comm->send_data->indices[i];
                G_recvbuf[idx] = result_func(G_recvbuf[idx], R_sendbuf[i]);
            }
            
            // Global communication... only send if compare yields true
            S_recvbuf = local_S_par_comm->recv_data->get_buffer<T>();
            std::fill(S_recvbuf.begin(), S_recvbuf.end(), 0);
            global_par_comm->conditional_comm_T<T, T>(G_recvbuf, 
                    S_recvbuf,
                    global_send_compares,  
                    global_recv_compares, 
                    compare_func, 
                    result_func);

            local_S_par_comm->communicate_T(S_recvbuf);
            S_sendbuf = local_S_par_comm->send_data->get_buffer<T>();

            local_L_par_comm->communicate_T(values);
            L_sendbuf = local_L_par_comm->send_data->get_buffer<T>();

            for (int i = 0; i < local_S_par_comm->send_data->size_msgs; i++)
            {
                idx = local_S_par_comm->send_data->indices[i];
                result[idx] = result_func(result[idx], S_sendbuf[i]);
            }
            for (int i = 0; i < local_L_par_comm->send_data->size_msgs; i++)
            {
                idx = local_L_par_comm->send_data->indices[i];
                result[idx] = result_func(result[idx], L_sendbuf[i]);
            }
        }


        // Helper Methods
        std::vector<double>& get_double_recv_buffer()
        {
            return recv_buffer;
        }
        std::vector<int>& get_int_recv_buffer()
        {
            return int_recv_buffer;
        }

        double get_comm_time()
        {
            return local_S_par_comm->get_comm_time() 
                + local_R_par_comm->get_comm_time()
                + local_L_par_comm->get_comm_time()
                + global_par_comm->get_comm_time();
        }

        int get_comm_n()
        {
            return local_S_par_comm->get_comm_n() 
                + local_R_par_comm->get_comm_n()
                + local_L_par_comm->get_comm_n()
                + global_par_comm->get_comm_n();
        }
        int get_comm_s()
        {
            return local_S_par_comm->get_comm_s() 
                + local_R_par_comm->get_comm_s()
                + local_L_par_comm->get_comm_s()
                + global_par_comm->get_comm_s();
        }


        // Class Attributes
        ParComm* local_S_par_comm;
        ParComm* local_R_par_comm;
        ParComm* local_L_par_comm;
        ParComm* global_par_comm;
        std::vector<double> recv_buffer;
        std::vector<int> int_recv_buffer;
    };
}
#endif
